# -*- coding: utf-8 -*-
"""
Диспетчер: событие Long Poll -> нужный обработчик.

Порядок проверок для входящего сообщения (как порядок роутеров в
Telegram-версии):

  1. payload кнопки — нижнее меню ({"cmd": …}) или inline-кнопка у
     клиента без callback-кнопок ({"a": …});
  2. команды (/todo текст, «Начать», «Помощь») — работают и тогда, когда
     бот ждёт текст исправления: переключение режима ожидание отменяет;
  3. ожидание исправления после 👎 — текст уходит в фидбэк;
  4. картинка во вложениях — распознавание;
  5. текст — перевод в текущем режиме.

В беседах бот отвечает только на команды, на упоминание
([club…|@бот] текст) и тому, от кого ждёт исправление: иначе он отвечал
бы на каждую реплику участников.

Что бы ни упало внутри обработчика, человек получает ответ, а не тишину.
"""

import logging
import re
from typing import Optional, Tuple

from .. import texts
from ..chat import Chat
from ..config import Config
from ..state import MODES, Corrections, valid_mode
from ..storage import Storage
from ..vk.api import VkApi
from ..vk.events import CALLBACK_CLIENT, ButtonPress, Message
from . import admin, common, feedback, ocr, settings, translate

log = logging.getLogger(__name__)

_COMMAND = re.compile(r"^/([\w-]+)(?:@\S+)?(?:\s+(.*))?$", re.S)


def parse_command(text: str) -> Optional[Tuple[Optional[str], str]]:
    """«/todo хальмг улс» -> ("todo", "хальмг улс"); незнакомая команда ->
    (None, ""); не команда -> None."""
    t = text.strip()
    if t.lower() in texts.BARE_COMMANDS:
        return texts.COMMAND_ALIASES[t.lower()], ""
    m = _COMMAND.match(t)
    if not m:
        return None
    return texts.COMMAND_ALIASES.get(m.group(1).lower()), (m.group(2) or "").strip()


class Dispatcher:
    def __init__(self, api: VkApi, storage: Storage, config: Config, group_id: int):
        self.api = api
        self.storage = storage
        self.config = config
        self.group_id = group_id
        self.corrections = Corrections()
        # обращение к боту в беседе: «[club123|@todorxoi], …»
        self._mention = re.compile(
            rf"^\s*\[(?:club|public){group_id}\|[^\]]*\][\s,:]*", re.I
        )

    # ---------------------------------------------------------- события

    async def feed(self, update: dict) -> None:
        kind = update.get("type")
        obj = update.get("object") or {}
        if kind == "message_new":
            msg = Message.parse(obj)
            chat = Chat(self.api, self.group_id, msg.peer_id, msg.from_id, msg.client)
            await self._guard(chat, self.on_message(chat, msg))
        elif kind == "message_event":
            press = ButtonPress.parse(obj)
            chat = Chat(
                self.api, self.group_id, press.peer_id, press.user_id, CALLBACK_CLIENT,
                event_id=press.event_id, cmid=press.conversation_message_id,
            )
            await self._guard(chat, self.on_action(chat, press.payload))

    async def _guard(self, chat: Chat, handler) -> None:
        try:
            await handler
        except Exception:
            log.exception("необработанная ошибка: peer=%s user=%s", chat.peer_id, chat.user_id)
            try:
                await chat.answer()
                await chat.send(texts.ERROR_GENERIC)
            except Exception:
                # не смогли даже ответить (бот заблокирован, сеть) — хватит лога
                log.warning("не удалось сообщить об ошибке в %s", chat.peer_id)
        finally:
            # нажатие callback-кнопки обязательно закрываем, иначе у
            # человека бесконечно крутится индикатор
            try:
                await chat.answer()
            except Exception:
                pass

    # ------------------------------------------------------- сообщения

    async def on_message(self, chat: Chat, msg: Message) -> None:
        if msg.from_id <= 0:
            return  # другие сообщества и боты в беседе

        text = msg.text
        addressed = not msg.is_chat
        if msg.is_chat:
            stripped = self._mention.sub("", text, count=1)
            addressed = stripped != text
            text = stripped

        payload = msg.payload or {}
        if payload.get("a"):
            await self.on_action(chat, payload)
            return
        if payload.get("cmd") or payload.get("command"):
            name = texts.COMMAND_ALIASES.get(str(payload.get("cmd") or payload.get("command")))
            await self.run_command(chat, msg, name, "")
            return

        command = parse_command(text)
        pending = self.corrections.pending(chat.peer_id, chat.user_id)
        if msg.is_chat and not (addressed or command or pending is not None):
            return

        if command:
            await self.run_command(chat, msg, *command)
            return

        if pending is not None:
            if text.strip():
                await feedback.save_correction(chat, text, self.storage, self.corrections)
            else:
                await chat.send(texts.FEEDBACK_NEED_TEXT)
            return

        picked = ocr.find_image(msg)
        if picked:
            await ocr.on_attachment(chat, picked, self.storage, self.config)
            return

        if text.strip():
            mode = valid_mode(await self.storage.get_mode(chat.user_id))
            await translate.handle_text(chat, text, mode, self.storage, self.config)
            return

        await chat.send(texts.NON_TEXT)

    async def run_command(self, chat: Chat, msg: Message, name: Optional[str], args: str) -> None:
        storage, config = self.storage, self.config
        if name is None:
            await chat.send(texts.UNKNOWN_COMMAND)
            return
        if name == "cancel":
            await feedback.cancel(chat, self.corrections)
            return

        # любая другая команда отменяет незаконченный ввод исправления
        self.corrections.clear(chat.peer_id, chat.user_id)
        if name == "start":
            await common.cmd_start(chat, storage, config)
        elif name == "help":
            await common.cmd_help(chat, storage, config)
        elif name == "mode":
            await common.cmd_mode(chat, storage)
        elif name in MODES:
            await common.switch(chat, name, args, storage, config)
        elif name == "ocr":
            await ocr.cmd_ocr(chat, msg, storage, config)
        elif name == "settings":
            await settings.cmd_settings(chat, storage)
        elif name == "stats":
            await admin.cmd_stats(chat, storage, config)
        elif name == "export":
            await admin.cmd_export(chat, args, storage, config)
        else:
            await chat.send(texts.UNKNOWN_COMMAND)

    # ---------------------------------------------------------- кнопки

    async def on_action(self, chat: Chat, payload: dict) -> None:
        action = payload.get("a")
        if action == "fb":
            await feedback.on_rating(chat, payload, self.storage, self.corrections)
        elif action == "conv":
            await translate.convert_further(chat, payload, self.storage, self.config)
        elif action == "set":
            await settings.on_settings(chat, payload, self.storage)
        else:
            await chat.answer()


__all__ = ["Dispatcher", "parse_command"]
