# -*- coding: utf-8 -*-
"""Обратная связь: 👍 / 👎 и приём правильного варианта после 👎.

Логика:
  • 👍 — сразу пишем оценку в БД и показываем плашку «спасибо»;
  • 👎 — пишем оценку и запоминаем, что ждём от этого человека
    правильный вариант (state.Corrections). Следующее текстовое сообщение
    уходит не в перевод, а в поле correction того же запроса;
  • /cancel — перестаём ждать, минус при этом остаётся: даже без
    правильного варианта информация «этот ответ плохой» полезна.

Кнопки под ответом после оценки не убираются (в Telegram-версии они
менялись на «оценено»): messages.edit в VK переписывает сообщение целиком,
вместе с текстом и вложениями. Повторное нажатие просто обновляет оценку.
"""

import logging

from .. import texts
from ..chat import Chat
from ..state import Corrections
from ..storage import Storage

log = logging.getLogger(__name__)


async def on_rating(
    chat: Chat, payload: dict, storage: Storage, corrections: Corrections
) -> None:
    rating = payload.get("v")
    request_id = payload.get("r")
    if rating not in ("up", "down") or not isinstance(request_id, int):
        await chat.answer()
        return

    row = await storage.get_request(request_id)
    if not row:
        await chat.notify(texts.UNKNOWN_REQUEST, alert=True)
        return

    await storage.save_feedback(
        request_id=request_id,
        user_id=chat.user_id,
        username=None,
        rating=rating,
    )

    if rating == "up":
        await chat.notify(texts.FEEDBACK_THANKS_UP)
        return

    await chat.answer()
    corrections.wait(chat.peer_id, chat.user_id, request_id)
    await chat.send(texts.FEEDBACK_ASK_CORRECTION)


async def cancel(chat: Chat, corrections: Corrections) -> None:
    if corrections.pending(chat.peer_id, chat.user_id) is None:
        await chat.send(texts.NOTHING_TO_CANCEL)
        return
    corrections.clear(chat.peer_id, chat.user_id)
    await chat.send(texts.FEEDBACK_CANCELLED)


async def save_correction(
    chat: Chat, text: str, storage: Storage, corrections: Corrections
) -> None:
    request_id = corrections.pending(chat.peer_id, chat.user_id)
    corrections.clear(chat.peer_id, chat.user_id)
    if request_id is None:
        await chat.send(texts.NOTHING_TO_CANCEL)
        return
    await storage.save_feedback(
        request_id=request_id,
        user_id=chat.user_id,
        username=None,
        rating="down",
        correction=text.strip(),
    )
    await chat.send(texts.FEEDBACK_SAVED)
