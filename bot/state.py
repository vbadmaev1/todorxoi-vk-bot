# -*- coding: utf-8 -*-
"""Режим и ожидание исправления.

Режим (translit | todo | image | fix) хранится в БД, в user_settings.mode,
и переживает перезапуск бота.

Ожидание исправления после 👎 — в памяти: если бот перезапустится, пока
человек пишет правильный вариант, его сообщение просто уйдёт в перевод.
Ради такой редкости заводить таблицу незачем. Ключ — (peer_id, user_id):
в беседе ждём ответа от того, кто нажал 👎, а не от любого участника.
"""

from typing import Dict, Optional, Tuple

DEFAULT_MODE = "translit"
MODES = ("translit", "todo", "image", "fix")


def valid_mode(mode: Optional[str]) -> str:
    return mode if mode in MODES else DEFAULT_MODE


class Corrections:
    def __init__(self):
        self._waiting: Dict[Tuple[int, int], int] = {}

    def wait(self, peer_id: int, user_id: int, request_id: int) -> None:
        self._waiting[(peer_id, user_id)] = request_id

    def pending(self, peer_id: int, user_id: int) -> Optional[int]:
        return self._waiting.get((peer_id, user_id))

    def clear(self, peer_id: int, user_id: int) -> None:
        self._waiting.pop((peer_id, user_id), None)
