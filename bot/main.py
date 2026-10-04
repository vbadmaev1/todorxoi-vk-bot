# -*- coding: utf-8 -*-
"""
Точка входа. Запуск:  python -m bot.main

Режим — Bots Long Poll: бот сам ходит к VK за событиями, домен и
сертификат не нужны. Экземпляр должен быть ровно один: два процесса с
одним ключом получат одни и те же события и ответят дважды.

Каждое событие обрабатывается отдельной задачей (до WORKERS сразу), чтобы
долгая картинка или распознавание фото не задерживали остальных.
"""

import asyncio
import logging
import sys

from .config import Config
from .handlers import Dispatcher
from .storage import Storage
from .vk import longpoll
from .vk.api import API_VERSION, VkApi, VkApiError

log = logging.getLogger("todorxoi")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def _warmup() -> None:
    """Прогреваем модели заранее: иначе первый же пользователь будет ждать
    загрузку чекпойнта (несколько секунд) прямо в чате."""
    from core.transliterate import model_info, warmup

    try:
        await asyncio.to_thread(warmup)
        log.info("модель прогрета: %s", model_info())
    except Exception:
        log.exception("прогрев модели не удался — бот продолжит работу")

    # словарь для текста без калмыцких букв (~35 МБ, ~1 с на загрузку)
    from core import fix_letters

    try:
        fixer = await asyncio.to_thread(fix_letters.get_fixer)
        log.info("словарь калмыцких букв готов: %d слов", len(fixer.freq))
    except Exception:
        log.exception("словарь калмыцких букв не загрузился — бот продолжит работу")

    # тодо бичиг/транслитерация -> кириллица: сегментатор и обратная модель
    from core import to_cyrillic

    try:
        info = await asyncio.to_thread(to_cyrillic.warmup)
        log.info("перевод в кириллицу готов: %s", info)
    except Exception:
        log.exception("модели кириллицы не загрузились — ответы будут без кириллицы")

    # Распознавание фото — отдельная модель и отдельные зависимости
    # (onnxruntime, scipy). Если их нет, бот работает, фото не читает.
    from core import ocr

    try:
        info = await asyncio.to_thread(ocr.warmup)
        log.info("распознавание фото готово: %s", info)
    except ocr.OcrUnavailable as exc:
        log.warning("распознавание фото недоступно: %s — текстовые режимы работают", exc)
    except Exception:
        log.exception("прогрев распознавания фото не удался — бот продолжит работу")


def _check_shaping() -> None:
    """Монгольское письмо курсивное: формы букв выбирает движок раскладки.
    Без него картинка выходит нечитаемой — проверяем на старте."""
    from core.todo_image import SHAPING_OK, shaping_status

    if SHAPING_OK:
        log.info("рендер картинок: %s", shaping_status())
    else:
        log.warning("КАРТИНКИ БУДУТ НЕПРАВИЛЬНЫМИ: %s", shaping_status())
        log.warning(
            "Починить: pip install -r requirements.txt — нужны uharfbuzz и "
            "freetype-py. До починки бот рисует картинки, но помечает их "
            "предупреждением; STRICT_SHAPING=1 выключает такой рендер совсем."
        )


async def _group(api: VkApi, config: Config) -> dict:
    """Сообщество, от имени которого работает ключ."""
    params = {"group_id": config.group_id} if config.group_id else {}
    resp = await api.call("groups.getById", **params)
    # 5.199 отдаёт {"groups": [...]}, старые версии — список
    groups = resp.get("groups") if isinstance(resp, dict) else resp
    if not groups:
        raise RuntimeError("VK не вернул сообщество для этого ключа")
    return groups[0]


async def _setup_group(api: VkApi, group_id: int) -> None:
    """Включить в сообществе то, без чего бот глух или беден:
      • Long Poll с нашей версией API и событиями message_new, message_event;
      • сообщения сообщества и «возможности ботов» (клавиатуры, кнопки,
        client_info), кнопку «Начать».
    Ключу нужно право «управление сообществом». Нет его — пишем, что
    включить руками, и работаем дальше: возможно, всё уже включено."""
    try:
        await api.call(
            "groups.setLongPollSettings",
            group_id=group_id, enabled=1, api_version=API_VERSION,
            message_new=1, message_event=1,
        )
        log.info("Long Poll включён: API %s, события message_new и message_event", API_VERSION)
    except VkApiError as exc:
        log.warning(
            "не удалось настроить Long Poll (%s). Включите вручную: Управление → "
            "Работа с API → Long Poll API: «Включён», версия %s; на вкладке «Типы "
            "событий» — «Входящее сообщение» и «Действие с сообщением».",
            exc, API_VERSION,
        )
    try:
        await api.call(
            "groups.setSettings",
            group_id=group_id, messages=1, bots_capabilities=1, bots_start_button=1,
        )
        log.info("сообщения сообщества и возможности ботов включены")
    except VkApiError as exc:
        log.warning(
            "не удалось включить сообщения и возможности ботов (%s). Включите "
            "вручную: Управление → Сообщения → «Включены»; там же → Настройки "
            "для бота → «Возможности ботов» и «Кнопка „Начать“».",
            exc,
        )


async def _check_longpoll(api: VkApi, group_id: int) -> None:
    """Самая частая причина «бот молчит»: в сообществе выключен Long Poll
    или не отмечено событие «Входящее сообщение». Ошибок тогда нет нигде —
    VK просто не присылает событий. Проверяем и пишем в лог прямо."""
    try:
        s = await api.call("groups.getLongPollSettings", group_id=group_id)
    except VkApiError as exc:
        log.warning("не удалось прочитать настройки Long Poll: %s", exc)
        return
    events = s.get("events") or {}
    missing = [
        title for key, title in (
            ("message_new", "«Входящее сообщение»"),
            ("message_event", "«Действие с сообщением»"),
        ) if not events.get(key)
    ]
    log.info(
        "Long Poll: %s, версия API %s, события: %s",
        "включён" if s.get("is_enabled") else "ВЫКЛЮЧЕН",
        s.get("api_version"),
        ", ".join(k for k, v in events.items() if v) or "никаких",
    )
    if not s.get("is_enabled"):
        log.error(
            "Long Poll в сообществе выключен — сообщения до бота не дойдут. "
            "Управление → Работа с API → Long Poll API → «Включён»."
        )
    if missing:
        log.error(
            "В Long Poll не отмечены события %s — бот их не получит. "
            "Управление → Работа с API → Long Poll API → Типы событий.",
            " и ".join(missing),
        )


async def _run(dispatcher: Dispatcher, api: VkApi, group_id: int, workers: int) -> None:
    sem = asyncio.Semaphore(workers)
    tasks = set()

    async def handle(update: dict) -> None:
        obj = update.get("object") or {}
        msg = obj.get("message") or obj
        log.info("событие %s от %s", update.get("type"), msg.get("from_id") or msg.get("user_id"))
        async with sem:
            await dispatcher.feed(update)

    while True:
        try:
            async for update in longpoll.listen(api, group_id):
                task = asyncio.create_task(handle(update))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except VkApiError as exc:
            if exc.code == 15 and exc.method == "groups.getLongPollServer":
                # права ключа не меняются — нужен новый ключ, перезапуск не поможет
                log.error(
                    "У ключа нет права «Управление сообществом» — без него VK не "
                    "даёт Long Poll. Создайте новый ключ: сообщество → Управление → "
                    "Работа с API → Ключи доступа → Создать ключ, отметьте "
                    "«Управление сообществом», «Сообщения сообщества», "
                    "«Фотографии», «Документы». Впишите его в VK_TOKEN и "
                    "перезапустите бота."
                )
                raise SystemExit(1)
            if exc.code in (5, 27):  # ключ неверный или отозван — перезапуск не поможет
                raise
            log.exception("long poll упал — перезапуск через 5 с")
            await asyncio.sleep(5)
        except Exception:
            log.exception("long poll упал — перезапуск через 5 с")
            await asyncio.sleep(5)


async def main() -> None:
    config = Config.from_env()
    setup_logging(config.log_level)
    config.validate()
    _check_shaping()

    storage = Storage(config.db_path, config.jsonl_path)
    storage.connect()
    api = VkApi(config.vk_token)

    try:
        try:
            group = await _group(api, config)
        except VkApiError as exc:
            if exc.code in (5, 27):
                log.error(
                    "VK не принял ключ (%s). Нужен ключ доступа СООБЩЕСТВА: "
                    "сообщество → Управление → Работа с API → Ключи доступа.", exc,
                )
            raise
        group_id = int(group["id"])
        log.info("сообщество: %s (id %s, vk.com/%s)", group.get("name"), group_id,
                 group.get("screen_name", f"club{group_id}"))
        if config.setup_longpoll:
            await _setup_group(api, group_id)
        await _check_longpoll(api, group_id)

        if config.warmup:
            await _warmup()

        dispatcher = Dispatcher(api, storage, config, group_id)
        log.info("бот запущен, жду сообщений")
        await _run(dispatcher, api, group_id, config.workers)
    finally:
        storage.close()
        await api.close()
        log.info("бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
