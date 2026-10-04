# -*- coding: utf-8 -*-
"""
Bots Long Poll API — аналог long polling в Telegram: бот сам ходит за
событиями, поэтому ни домена, ни сертификата не нужно.

    groups.getLongPollServer -> {server, key, ts}
    GET {server}?act=a_check&key=…&ts=…&wait=25 -> {ts, updates: [...]}

Сервер может ответить {"failed": N}:
  1 — история событий устарела, продолжить с присланного ts;
  2 — истёк ключ, взять новый;
  3 — потеряна информация, взять новые ключ и ts.

ts при старте берётся свежий, поэтому события, накопившиеся, пока бот
лежал, не обрабатываются — как drop_pending_updates в Telegram-версии.
"""

import asyncio
import logging
from typing import AsyncIterator

import aiohttp

from .api import VkApi

log = logging.getLogger(__name__)


async def listen(api: VkApi, group_id: int, wait: int = 25) -> AsyncIterator[dict]:
    server = await api.call("groups.getLongPollServer", group_id=group_id)
    ts = server["ts"]
    timeout = aiohttp.ClientTimeout(total=wait + 15)
    backoff = 1.0
    while True:
        params = {"act": "a_check", "key": server["key"], "ts": ts, "wait": wait}
        try:
            async with api.session.get(
                server["server"], params=params, timeout=timeout
            ) as resp:
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            log.warning("long poll: %s, повтор через %.0f с", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        backoff = 1.0

        failed = data.get("failed")
        if failed == 1:
            ts = data["ts"]
            continue
        if failed in (2, 3):
            fresh = await api.call("groups.getLongPollServer", group_id=group_id)
            server["key"], server["server"] = fresh["key"], fresh["server"]
            if failed == 3:
                ts = fresh["ts"]
            continue
        if failed:
            log.warning("long poll: неизвестный ответ %s", data)
            await asyncio.sleep(1)
            continue

        ts = data.get("ts", ts)
        for update in data.get("updates", []):
            yield update
