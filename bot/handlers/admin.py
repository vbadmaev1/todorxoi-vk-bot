# -*- coding: utf-8 -*-
"""Админские команды: статистика и выгрузка собранного фидбэка.

Доступны только тем, чьи id ВКонтакте перечислены в ADMIN_IDS. Пока эта
переменная пустая, команды не работают ни у кого — чтобы выгрузку не мог
скачать случайный пользователь.
"""

import logging

from .. import texts
from ..chat import Chat
from ..config import Config
from ..formatting import fmt_ms
from ..storage import Storage

log = logging.getLogger(__name__)


async def cmd_stats(chat: Chat, storage: Storage, config: Config) -> None:
    if not config.is_admin(chat.user_id):
        await chat.send(texts.NOT_ADMIN)
        return
    s = await storage.stats()
    by_target = ", ".join(f"{k}: {v}" for k, v in sorted(s["by_target"].items())) or "—"
    total_votes = s["up"] + s["down"]
    share = f"{s['up'] / total_votes * 100:.0f}%" if total_votes else "—"
    await chat.send(
        texts.ADMIN_STATS.format(
            requests=s["requests"],
            errors=s["errors"],
            by_target=by_target,
            users=s["users"],
            avg=fmt_ms(s["avg_ms"]),
            up=s["up"],
            down=s["down"],
            share=share,
            corrections=s["corrections"],
        )
    )


async def cmd_export(chat: Chat, args: str, storage: Storage, config: Config) -> None:
    """/export — весь фидбэк; /export corrections — только исправления."""
    if not config.is_admin(chat.user_id):
        await chat.send(texts.NOT_ADMIN)
        return
    only_corrections = (args or "").strip().lower().startswith("corr")
    data = await storage.export_csv(only_corrections=only_corrections)
    name = "corrections.csv" if only_corrections else "feedback.csv"
    doc = await chat.api.upload_doc(chat.peer_id, data, name)
    await chat.send(
        texts.ADMIN_EXPORT_CORRECTIONS if only_corrections else texts.ADMIN_EXPORT_ALL,
        attachments=[doc],
    )
