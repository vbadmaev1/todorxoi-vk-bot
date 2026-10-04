# -*- coding: utf-8 -*-
"""Начать / помощь / режимы."""

import logging
from typing import Optional

from core import MAX_INPUT_CHARS

from .. import keyboards, texts
from ..chat import Chat
from ..config import Config
from ..state import valid_mode
from ..storage import Storage
from .translate import handle_text

log = logging.getLogger(__name__)


def _news(config: Config) -> str:
    return texts.NEWS.format(url=config.news_url) if config.news_url else ""


async def cmd_start(chat: Chat, storage: Storage, config: Config) -> None:
    mode = valid_mode(await storage.get_mode(chat.user_id))
    await chat.send(
        texts.START.format(mode=texts.MODE_TITLES[mode]),
        keyboards.main_menu(chat.client, mode),
    )


async def cmd_help(chat: Chat, storage: Storage, config: Config) -> None:
    mode = valid_mode(await storage.get_mode(chat.user_id))
    await chat.send(
        texts.HELP.format(max_chars=MAX_INPUT_CHARS, news=_news(config)),
        keyboards.main_menu(chat.client, mode),
    )


async def cmd_mode(chat: Chat, storage: Storage) -> None:
    mode = valid_mode(await storage.get_mode(chat.user_id))
    await chat.send(texts.MODE_CURRENT.format(mode=texts.MODE_TITLES[mode]))


async def switch(
    chat: Chat, mode: str, payload: Optional[str], storage: Storage, config: Config
) -> None:
    await storage.set_mode(chat.user_id, mode)
    if payload and payload.strip():
        # текст пришёл сразу вместе с командой — обрабатываем, не переспрашивая
        await handle_text(chat, payload.strip(), mode, storage, config)
        return
    await chat.send(
        texts.MODE_SWITCHED.format(mode=texts.MODE_TITLES[mode]),
        keyboards.main_menu(chat.client, mode),
    )
