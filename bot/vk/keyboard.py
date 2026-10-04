# -*- coding: utf-8 -*-
"""
Клавиатуры VK — JSON в параметре keyboard у messages.send/edit.

Лимиты (их VK проверяет и при нарушении не отправляет сообщение вовсе):
  • обычная клавиатура — до 10 рядов, до 5 кнопок в ряду, всего до 40;
  • inline (под сообщением) — до 6 рядов, до 5 в ряду и всего до 10;
  • подпись кнопки — до 40 символов, payload — до 255 байт.

Цвета есть только у text- и callback-кнопок: primary (синяя),
secondary (белая), positive (зелёная), negative (красная).
"""

import json
from typing import List, Optional

LABEL_LIMIT = 40
PAYLOAD_LIMIT = 255
INLINE_MAX_ROWS = 6
INLINE_MAX_BUTTONS = 10

TEXT = "text"
CALLBACK = "callback"


def button(label: str, payload: dict, kind: str = CALLBACK, color: Optional[str] = None) -> dict:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(raw.encode("utf-8")) > PAYLOAD_LIMIT:
        raise ValueError(f"payload длиннее {PAYLOAD_LIMIT} байт: {raw}")
    if len(label) > LABEL_LIMIT:
        label = label[: LABEL_LIMIT - 1] + "…"
    out = {"action": {"type": kind, "label": label, "payload": raw}}
    if color:
        out["color"] = color
    return out


def keyboard(rows: List[List[dict]], inline: bool = False, one_time: bool = False) -> str:
    rows = [row for row in rows if row]
    if inline:
        total = sum(len(row) for row in rows)
        if len(rows) > INLINE_MAX_ROWS or total > INLINE_MAX_BUTTONS:
            raise ValueError(
                f"inline-клавиатура {len(rows)} рядов / {total} кнопок — больше лимита VK"
            )
    body = {"buttons": rows, "inline": inline}
    if not inline:
        body["one_time"] = one_time
    return json.dumps(body, ensure_ascii=False)


EMPTY = json.dumps({"buttons": [], "one_time": True})
