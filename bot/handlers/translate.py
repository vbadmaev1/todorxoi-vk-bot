# -*- coding: utf-8 -*-
"""Три основных сценария: транслитерация, тодо бичиг, картинка (+ исправление).

Тяжёлая часть (модель + рендер) уезжает в поток через asyncio.to_thread,
иначе один длинный запрос подвесил бы всех остальных пользователей.
"""

import asyncio
import json
import logging

import core

from .. import formatting, keyboards, texts
from ..chat import Chat, batched
from ..config import Config
from ..storage import Storage
from .settings import load_fix_letters, load_options, load_punctuation, load_show_time

log = logging.getLogger(__name__)

# соотношение сторон, после которого VK не примет картинку как фото (1:20)
PHOTO_MAX_RATIO = 19
# сумма сторон фото у VK — не больше 14000
PHOTO_MAX_SIDES = 14000
# сколько листов грузить в VK одновременно: быстрее, но в рамках 20 вызовов/с
_UPLOADS_AT_ONCE = 3


async def handle_text(
    chat: Chat, text: str, target: str, storage: Storage, config: Config
) -> None:
    """Общая точка: посчитать, записать в БД, ответить с кнопками."""
    await chat.typing()

    user_id = chat.user_id
    base = {
        "user_id": user_id,
        "username": None,
        "chat_id": chat.peer_id,
        "target": target,
        "input_text": text,
    }

    options = await load_options(storage, user_id) if target == core.TARGET_IMAGE else None
    fix_letters = await load_fix_letters(storage, user_id)
    punctuation = await load_punctuation(storage, user_id)

    try:
        res = await asyncio.to_thread(
            core.process, text, target, options, fix_letters, punctuation
        )
    except core.PipelineError as exc:
        await storage.save_request(ok=False, error=str(exc), **base)
        await chat.send(f"⚠️ {exc}")
        return
    except Exception:
        log.exception("сбой обработки: target=%s text=%r", target, text[:80])
        await storage.save_request(ok=False, error="internal", **base)
        await chat.send(texts.ERROR_GENERIC)
        return

    request_id = await storage.save_request(
        source_script=res.source_script,
        translit=res.translit,
        todo=res.todo,
        cyrillic=res.cyrillic,
        elapsed_ms=res.elapsed_ms,
        steps=res.steps_ms,
        ok=True,
        letters=_letters_log(res, fix_letters),
        **base,
    )

    with_feedback = target != core.TARGET_TRANSLIT or config.feedback_on_translit
    markup = keyboards.result_keyboard(chat.client, request_id, target, with_feedback)

    show_time = await load_show_time(storage, user_id)
    if target == core.TARGET_IMAGE:
        await send_image(chat, res, markup, show_time)
    else:
        await chat.send(formatting.render_result(res, show_time), markup)


def _letters_log(res, enabled: bool):
    """Что записать в БД про калмыцкие буквы: по этим записям потом
    калибруются пороги core/fix_letters.py на настоящих запросах."""
    if res.target == core.TARGET_FIX:
        # исправленный текст нужен кнопкам «→ Транслитерация» и т.д.
        return {"enabled": True, "flag": res.letters_flag, "fixes": res.letter_fixes,
                "text": res.fixed_text}
    if not (res.letters_flag or res.letter_fixes):
        return None
    return {"enabled": enabled, "flag": res.letters_flag, "fixes": res.letter_fixes}


def photo_ok(size) -> bool:
    """Примет ли VK картинку такого размера как фото."""
    w, h = size or (0, 0)
    if not (w and h):
        return True
    return max(w, h) / min(w, h) <= PHOTO_MAX_RATIO and w + h <= PHOTO_MAX_SIDES


async def send_image(chat: Chat, res, markup, show_time: bool = True) -> None:
    """Отправляет листы картинки.

    В VK к сообщению прикрепляется до 10 вложений, поэтому длинный текст
    приходит пачками по 10 листов (в Telegram — по листу на сообщение).
    Кнопки — под последней пачкой: они относятся ко всему ответу.

    Прозрачный фон — документом: фото VK пережимает в JPEG, прозрачность
    стала бы чёрным прямоугольником. Слишком вытянутая картинка — тоже
    документом: как фото VK её не примет.
    """
    total = len(res.pages)
    sem = asyncio.Semaphore(_UPLOADS_AT_ONCE)

    async def upload(number: int, buf, size) -> str:
        name = f"todo_bichig_{number}.png" if total > 1 else "todo_bichig.png"
        as_doc = res.transparent or not photo_ok(size)
        async with sem:
            return await chat.upload(buf.getvalue(), name, as_doc=as_doc)

    await chat.typing()
    refs = await asyncio.gather(
        *(upload(n, buf, size) for n, (buf, size) in enumerate(res.pages, start=1))
    )
    chunks = batched(refs)
    first = 1
    for k, chunk in enumerate(chunks):
        last_chunk = k == len(chunks) - 1
        caption = formatting.image_caption(
            res, first, first + len(chunk) - 1, total,
            show_time=show_time, final=last_chunk,
        )
        await chat.send(caption, markup if last_chunk else None, attachments=chunk)
        first += len(chunk)


async def convert_further(
    chat: Chat, payload: dict, storage: Storage, config: Config
) -> None:
    """Кнопки «→ Тодо бичиг» и «→ Картинка» под готовым ответом."""
    target = payload.get("t")
    request_id = payload.get("r")
    if target not in (core.TARGET_TRANSLIT, core.TARGET_TODO, core.TARGET_IMAGE) or not isinstance(
        request_id, int
    ):
        await chat.answer()
        return

    row = await storage.get_request(request_id)
    if not row:
        await chat.notify(texts.UNKNOWN_REQUEST, alert=True)
        return

    await chat.answer()
    await handle_text(chat, source_text(row), target, storage, config)


def source_text(row: dict) -> str:
    """Что переводить по кнопке под ответом: для режима исправления — уже
    исправленный текст, для остальных — то, что прислал человек."""
    if row.get("target") == core.TARGET_FIX and row.get("letters_json"):
        try:
            return json.loads(row["letters_json"]).get("text") or row["input_text"]
        except ValueError:
            pass
    return row["input_text"]
