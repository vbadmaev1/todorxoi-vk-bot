# -*- coding: utf-8 -*-
"""
События Long Poll, которые нужны боту, в виде простых объектов.

  message_new   — входящее сообщение. Нажатие обычной (текстовой) кнопки
                  тоже приходит так: текст — подпись кнопки, в payload —
                  то, что в неё зашили.
  message_event — нажатие callback-кнопки. Сообщения в чате при этом не
                  появляется, а клиент крутит индикатор, пока бот не
                  ответит messages.sendMessageEventAnswer.

Что умеет клиент пользователя (клавиатуры, inline, callback-кнопки),
VK сообщает в client_info каждого message_new: по нему решается, какие
кнопки слать.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

# peer_id бесед начинается отсюда; меньше — личка с пользователем
CHAT_PEER_START = 2_000_000_000


def _payload(raw) -> Optional[dict]:
    """payload в message_new — JSON-строка, в message_event — уже объект."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class ClientInfo:
    button_actions: frozenset = frozenset({"text"})
    keyboard: bool = True
    inline_keyboard: bool = True

    @property
    def callback(self) -> bool:
        return "callback" in self.button_actions

    @classmethod
    def parse(cls, raw: Optional[dict]) -> "ClientInfo":
        if not raw:
            return cls()
        return cls(
            button_actions=frozenset(raw.get("button_actions") or ("text",)),
            keyboard=bool(raw.get("keyboard", True)),
            inline_keyboard=bool(raw.get("inline_keyboard", True)),
        )


# Нажатие callback-кнопки пришло от клиента, который их умеет
CALLBACK_CLIENT = ClientInfo(
    button_actions=frozenset({"text", "callback"}), keyboard=True, inline_keyboard=True
)


@dataclass
class Message:
    peer_id: int
    from_id: int
    text: str = ""
    payload: Optional[dict] = None
    attachments: list = field(default_factory=list)
    reply_message: Optional[dict] = None
    fwd_messages: list = field(default_factory=list)
    conversation_message_id: Optional[int] = None
    client: ClientInfo = field(default_factory=ClientInfo)

    @property
    def is_chat(self) -> bool:
        return self.peer_id >= CHAT_PEER_START

    @classmethod
    def parse(cls, obj: dict) -> "Message":
        # с версии 5.103 сообщение лежит в object.message, рядом client_info
        msg = obj.get("message", obj)
        return cls(
            peer_id=int(msg.get("peer_id") or msg.get("from_id")),
            from_id=int(msg.get("from_id", 0)),
            text=msg.get("text") or "",
            payload=_payload(msg.get("payload")),
            attachments=msg.get("attachments") or [],
            reply_message=msg.get("reply_message"),
            fwd_messages=msg.get("fwd_messages") or [],
            conversation_message_id=msg.get("conversation_message_id"),
            client=ClientInfo.parse(obj.get("client_info")),
        )


@dataclass
class ButtonPress:
    peer_id: int
    user_id: int
    event_id: str
    payload: dict
    conversation_message_id: Optional[int] = None

    @classmethod
    def parse(cls, obj: dict) -> "ButtonPress":
        return cls(
            peer_id=int(obj["peer_id"]),
            user_id=int(obj["user_id"]),
            event_id=str(obj["event_id"]),
            payload=_payload(obj.get("payload")) or {},
            conversation_message_id=obj.get("conversation_message_id"),
        )
