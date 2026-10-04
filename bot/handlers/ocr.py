# -*- coding: utf-8 -*-
"""Фото → текст: распознавание тодо бичиг (core/ocr.py).

Фото принимается в любом режиме: режимы касаются того, во что превращать
текст, а картинку ни с чем не спутать. Команда /ocr — подсказка; с фото
в том же сообщении, ответом на сообщение с фото или с пересланным фото —
распознаёт его. Картинка, присланная файлом (документом), тоже годится —
её VK не пережимает.

Ответ: картинка с рамками и номерами столбцов (как бот разрезал страницу)
и в том же сообщении кириллица, транслитерация и юникод тодо бичиг по
строке на столбец, под ними 👍/👎. В VK подпись к картинке — обычный текст
сообщения до 4096 символов, так что почти всегда это одно сообщение.

Распознавание — около секунды счётной работы и ~160 МБ памяти на страницу:
  • уезжает в поток (asyncio.to_thread), чтобы бот не вставал для всех;
  • идёт по одной картинке за раз (_lock): две страницы параллельно удвоили
    бы пик памяти, а быстрее на сервере с одним-двумя ядрами не стало бы.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from core import ocr as core_ocr

from .. import formatting, keyboards, texts
from ..chat import Chat
from ..config import Config
from ..storage import Storage
from ..vk.events import Message
from .settings import load_punctuation, load_show_time
from .translate import photo_ok

log = logging.getLogger(__name__)

TARGET_OCR = "ocr"
# больше не качаем: core/ocr.py всё равно ужимает картинку до 2500 px
MAX_FILE_BYTES = 20 * 1024 * 1024

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "heic"}

_lock = asyncio.Lock()


@dataclass
class Picked:
    """Что нашлось во вложениях: картинка (url) или что-то другое (kind)."""
    kind: str                    # image | pdf | other
    url: str = ""
    size: Optional[int] = None
    ref: str = ""                # «photo<owner>_<id>» — для записи в БД


def _photo(photo: dict) -> Picked:
    # orig_photo — оригинал без пережатия (есть в новых версиях API);
    # иначе — самый крупный из вариантов sizes
    orig = photo.get("orig_photo") or {}
    if orig.get("url"):
        url = orig["url"]
    else:
        sizes = photo.get("sizes") or []
        best = max(sizes, key=lambda s: (s.get("width", 0) * s.get("height", 0)), default={})
        url = best.get("url", "")
    ref = f"photo{photo.get('owner_id')}_{photo.get('id')}"
    return Picked("image", url=url, ref=ref)


def _doc(doc: dict) -> Picked:
    ext = (doc.get("ext") or "").lower()
    if ext in IMAGE_EXTS:
        return Picked(
            "image", url=doc.get("url", ""), size=doc.get("size"),
            ref=f"doc{doc.get('owner_id')}_{doc.get('id')}",
        )
    return Picked("pdf" if ext == "pdf" else "other")


def _from_attachments(attachments: list) -> Optional[Picked]:
    """Первая картинка из вложений; если картинок нет, но есть файл — что за файл."""
    other = None
    for att in attachments or []:
        kind = att.get("type")
        if kind == "photo":
            return _photo(att["photo"])
        if kind == "doc":
            picked = _doc(att["doc"])
            if picked.kind == "image":
                return picked
            other = other or picked
    return other


def find_image(msg: Message, look_around: bool = False) -> Optional[Picked]:
    """Картинка из сообщения. look_around — для /ocr: ещё в сообщении, на
    которое ответили, и в пересланных."""
    picked = _from_attachments(msg.attachments)
    if picked and picked.kind == "image":
        return picked
    if look_around:
        nearby = ([msg.reply_message] if msg.reply_message else []) + list(msg.fwd_messages)
        for item in nearby:
            found = _from_attachments(item.get("attachments") or [])
            if found and found.kind == "image":
                return found
    return picked


async def cmd_ocr(chat: Chat, msg: Message, storage: Storage, config: Config) -> None:
    picked = find_image(msg, look_around=True)
    if picked and picked.kind == "image":
        await read_image(chat, picked, storage, config)
    else:
        await chat.send(texts.OCR_HINT)


async def on_attachment(chat: Chat, picked: Picked, storage: Storage, config: Config) -> None:
    if picked.kind == "image":
        await read_image(chat, picked, storage, config)
    elif picked.kind == "pdf":
        await chat.send(texts.OCR_PDF)
    else:
        await chat.send(texts.OCR_NOT_IMAGE)


async def read_image(chat: Chat, picked: Picked, storage: Storage, config: Config) -> None:
    if picked.size and picked.size > MAX_FILE_BYTES:
        await chat.send(texts.OCR_TOO_BIG)
        return

    base = {
        "user_id": chat.user_id,
        "username": None,
        "chat_id": chat.peer_id,
        "target": TARGET_OCR,
        "source_script": "photo",
        # саму картинку не храним: по ссылке её можно скачать заново, пока
        # VK её не удалил, а по ref — найти вложение через API
        "input_text": f"{picked.ref} {picked.url}".strip(),
    }

    punctuation = await load_punctuation(storage, chat.user_id)
    await chat.typing()
    try:
        data = await chat.api.download(picked.url, MAX_FILE_BYTES)
    except ValueError:
        await chat.send(texts.OCR_TOO_BIG)
        return
    except Exception as exc:
        log.warning("не скачалась картинка %s: %s", picked.ref, exc)
        await storage.save_request(ok=False, error=f"download: {exc}", **base)
        await chat.send(texts.OCR_DOWNLOAD_FAILED)
        return

    try:
        async with _lock:
            res = await asyncio.to_thread(
                core_ocr.recognize, data, config.ocr_overlay, punctuation
            )
    except core_ocr.OcrUnavailable as exc:
        log.warning("распознавание фото недоступно: %s", exc)
        await storage.save_request(ok=False, error=f"ocr unavailable: {exc}", **base)
        await chat.send(texts.OCR_UNAVAILABLE)
        return
    except core_ocr.OcrError as exc:
        await storage.save_request(ok=False, error=str(exc), **base)
        await chat.send(f"⚠️ {exc}")
        return
    except Exception:
        log.exception("сбой распознавания фото: %s", picked.ref)
        await storage.save_request(ok=False, error="internal", **base)
        await chat.send(texts.ERROR_GENERIC)
        return

    request_id = await storage.save_request(
        translit=res.translit,
        todo=res.todo,
        cyrillic=res.cyrillic or None,
        elapsed_ms=res.elapsed_ms,
        steps=res.steps_ms,
        ok=True,
        **base,
    )
    markup = keyboards.result_keyboard(chat.client, request_id, TARGET_OCR)
    show_time = await load_show_time(storage, chat.user_id)
    await _answer(chat, res, markup, show_time)


async def _answer(chat: Chat, res, markup, show_time: bool = True) -> None:
    parts = formatting.render_ocr(res, show_time)
    attachments = []
    if res.overlay:
        # один столбец — картинка узкая и высокая; такую VK как фото не примет
        as_doc = not photo_ok(res.overlay_size)
        attachments = [await chat.upload(res.overlay, "columns.jpg", as_doc=as_doc)]
    if len(parts) == 1:
        await chat.send(parts[0], markup, attachments=attachments)
        return
    # длинный ответ: картинка с первой частью, кнопки — под последней
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        await chat.send(
            part, markup if last else None, attachments=attachments if k == 0 else ()
        )
