# -*- coding: utf-8 -*-
"""
storage.py — куда бот складывает запросы и обратную связь.

Два слоя, намеренно дублирующих друг друга:

  1. SQLite (data/todorxoi.sqlite3) — основное хранилище. Две таблицы:
     requests (что попросили, что получилось, сколько это заняло) и
     feedback (палец вверх/вниз и присланный правильный ответ). Из него
     удобно делать выгрузку и считать статистику.

  2. JSONL (data/events.jsonl) — append-only лог, по одной json-строке на
     событие. Нужен на случай, если файл БД потеряется при переезде/
     пересборке контейнера: восстановить из него можно всё.

ВАЖНО про хостинг: и БД, и лог — обычные файлы в папке data/. Если бот
крутится в Docker/на PaaS с эфемерной файловой системой, эту папку надо
примонтировать как volume, иначе весь собранный фидбэк исчезнет при
рестарте. Стандартные «логи хостинга» этого не заменяют: они ротируются
и не предназначены для того, чтобы потом строить из них датасет.

Все обращения синхронные (sqlite3 + одна блокировка), а наружу торчат
async-обёртки через asyncio.to_thread — чтобы не блокировать цикл событий.

Схема та же, что у Telegram-версии, — выгрузки и tools/restore_from_jsonl.py
одинаковые. user_id и chat_id здесь — id ВКонтакте (chat_id — peer_id
диалога), username пустой: в событиях VK его нет. Плюс колонка mode в
user_settings: выбранный режим переживает перезапуск бота.
"""

import asyncio
import csv
import io
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    user_id      INTEGER,
    username     TEXT,
    chat_id      INTEGER,
    target       TEXT    NOT NULL,   -- translit | todo | image | fix | ocr
    source_script TEXT,              -- cyrillic | translit | todo
    input_text   TEXT    NOT NULL,
    translit     TEXT,
    todo         TEXT,
    cyrillic     TEXT,               -- кириллица из тодо бичиг/транслитерации (фото, латиница)
    elapsed_ms   REAL,
    steps_json   TEXT,
    ok           INTEGER NOT NULL DEFAULT 1,
    error        TEXT,
    letters_json TEXT                -- кириллица без ә ө ү һ җ ң: что заметили/исправили
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  INTEGER NOT NULL REFERENCES requests(id),
    created_at  TEXT    NOT NULL,
    user_id     INTEGER,
    username    TEXT,
    rating      TEXT    NOT NULL,    -- up | down
    correction  TEXT
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id    INTEGER PRIMARY KEY,
    fg         TEXT,
    bg         TEXT,
    size       TEXT,
    font       TEXT,
    fix_letters TEXT,               -- on | off: исправлять текст без калмыцких букв
    punctuation TEXT,               -- off | frame | all: знаки препинания тодо бичиг
    show_time  TEXT,                -- on | off: время работы под ответом
    mode       TEXT,                -- translit | todo | image | fix: текущий режим
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback(request_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique
    ON feedback(request_id, user_id);
"""


_SETTING_FIELDS = (
    "fg", "bg", "size", "font", "fix_letters", "punctuation", "show_time", "mode",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path, jsonl_path: Path):
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ---------------------------------------------------------------- setup

    # старые базы заводились без этих колонок — добавляем на лету
    _MIGRATIONS = (
        "ALTER TABLE user_settings ADD COLUMN font TEXT",
        "ALTER TABLE user_settings ADD COLUMN fix_letters TEXT",
        "ALTER TABLE requests ADD COLUMN letters_json TEXT",
        "ALTER TABLE user_settings ADD COLUMN punctuation TEXT",
        "ALTER TABLE requests ADD COLUMN cyrillic TEXT",
        "ALTER TABLE user_settings ADD COLUMN show_time TEXT",
        "ALTER TABLE user_settings ADD COLUMN mode TEXT",
    )

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        for sql in self._MIGRATIONS:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # колонка уже есть — обычное дело
        self._conn.commit()
        log.info("БД готова: %s", self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ внутреннее

    def _append_jsonl(self, payload: dict) -> None:
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # лог — вспомогательный слой, из-за него бот падать не должен
            log.exception("не удалось записать в %s", self.jsonl_path)

    # --------------------------------------------------------------- запись

    def _save_request(self, **kw) -> int:
        row = {
            "created_at": _now(),
            "user_id": kw.get("user_id"),
            "username": kw.get("username"),
            "chat_id": kw.get("chat_id"),
            "target": kw["target"],
            "source_script": kw.get("source_script"),
            "input_text": kw["input_text"],
            "translit": kw.get("translit"),
            "todo": kw.get("todo"),
            "cyrillic": kw.get("cyrillic"),
            "elapsed_ms": kw.get("elapsed_ms"),
            "steps_json": json.dumps(kw.get("steps") or {}, ensure_ascii=False),
            "ok": 1 if kw.get("ok", True) else 0,
            "error": kw.get("error"),
            "letters_json": (
                json.dumps(kw["letters"], ensure_ascii=False) if kw.get("letters") else None
            ),
        }
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO requests
                   (created_at, user_id, username, chat_id, target, source_script,
                    input_text, translit, todo, cyrillic, elapsed_ms, steps_json, ok,
                    error, letters_json)
                   VALUES (:created_at, :user_id, :username, :chat_id, :target,
                           :source_script, :input_text, :translit, :todo, :cyrillic,
                           :elapsed_ms, :steps_json, :ok, :error, :letters_json)""",
                row,
            )
            self._conn.commit()
            request_id = cur.lastrowid
        self._append_jsonl({"event": "request", "id": request_id, **row})
        return request_id

    def _save_feedback(self, **kw) -> None:
        row = {
            "request_id": kw["request_id"],
            "created_at": _now(),
            "user_id": kw.get("user_id"),
            "username": kw.get("username"),
            "rating": kw["rating"],
            "correction": kw.get("correction"),
        }
        with self._lock:
            # один пользователь — одна оценка на запрос; повторный клик
            # просто перезаписывает предыдущую
            self._conn.execute(
                """INSERT INTO feedback
                     (request_id, created_at, user_id, username, rating, correction)
                   VALUES (:request_id, :created_at, :user_id, :username,
                           :rating, :correction)
                   ON CONFLICT(request_id, user_id) DO UPDATE SET
                     rating = excluded.rating,
                     created_at = excluded.created_at,
                     correction = COALESCE(excluded.correction, feedback.correction)""",
                row,
            )
            self._conn.commit()
        self._append_jsonl({"event": "feedback", **row})

    def _get_request(self, request_id: int) -> Optional[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM requests WHERE id = ?", (request_id,)
            )
            row = cur.fetchone()
        return dict(row) if row else None

    # ---------------------------------------------- настройки пользователя

    def _get_settings(self, user_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_SETTING_FIELDS)} FROM user_settings "
                "WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return {}
        # None-поля не отдаём: пусть сработает значение по умолчанию
        return {k: row[k] for k in _SETTING_FIELDS if row[k]}

    def _set_setting(self, user_id: int, field: str, value: str) -> None:
        if field not in _SETTING_FIELDS:
            raise ValueError(f"неизвестная настройка: {field}")
        with self._lock:
            self._conn.execute(
                f"""INSERT INTO user_settings (user_id, {field}, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                      {field} = excluded.{field},
                      updated_at = excluded.updated_at""",
                (user_id, value, _now()),
            )
            self._conn.commit()

    def _reset_settings(self, user_id: int) -> None:
        # «сбросить всё» — про настройки, выбранный режим остаётся
        cleared = ", ".join(f"{f} = NULL" for f in _SETTING_FIELDS if f != "mode")
        with self._lock:
            self._conn.execute(
                f"UPDATE user_settings SET {cleared}, updated_at = ? WHERE user_id = ?",
                (_now(), user_id),
            )
            self._conn.commit()

    def _stats(self) -> dict:
        with self._lock:
            c = self._conn
            total = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            users = c.execute(
                "SELECT COUNT(DISTINCT user_id) FROM requests"
            ).fetchone()[0]
            by_target = dict(
                c.execute(
                    "SELECT target, COUNT(*) FROM requests GROUP BY target"
                ).fetchall()
            )
            errors = c.execute(
                "SELECT COUNT(*) FROM requests WHERE ok = 0"
            ).fetchone()[0]
            avg_ms = c.execute(
                "SELECT AVG(elapsed_ms) FROM requests WHERE ok = 1"
            ).fetchone()[0]
            up = c.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 'up'"
            ).fetchone()[0]
            down = c.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 'down'"
            ).fetchone()[0]
            corrections = c.execute(
                "SELECT COUNT(*) FROM feedback WHERE correction IS NOT NULL "
                "AND correction != ''"
            ).fetchone()[0]
        return {
            "requests": total,
            "users": users,
            "by_target": by_target,
            "errors": errors,
            "avg_ms": avg_ms or 0.0,
            "up": up,
            "down": down,
            "corrections": corrections,
        }

    def _export_csv(self, only_corrections: bool = False) -> bytes:
        query = """
            SELECT f.id            AS feedback_id,
                   f.created_at    AS feedback_at,
                   f.rating        AS rating,
                   f.correction    AS correction,
                   f.user_id       AS feedback_user_id,
                   r.id            AS request_id,
                   r.created_at    AS request_at,
                   r.target        AS target,
                   r.source_script AS source_script,
                   r.input_text    AS input_text,
                   r.translit      AS bot_translit,
                   r.todo          AS bot_todo,
                   r.elapsed_ms    AS elapsed_ms,
                   r.letters_json  AS letters
            FROM feedback f
            JOIN requests r ON r.id = f.request_id
        """
        if only_corrections:
            query += " WHERE f.correction IS NOT NULL AND f.correction != ''"
        query += " ORDER BY f.id"
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        if rows:
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(list(row))
        else:
            writer.writerow(["feedback_id"])
        return buf.getvalue().encode("utf-8-sig")

    # ------------------------------------------------------- async-обёртки

    async def save_request(self, **kw) -> int:
        return await asyncio.to_thread(self._save_request, **kw)

    async def save_feedback(self, **kw) -> None:
        await asyncio.to_thread(self._save_feedback, **kw)

    async def get_request(self, request_id: int) -> Optional[dict]:
        return await asyncio.to_thread(self._get_request, request_id)

    async def get_settings(self, user_id: int) -> dict:
        return await asyncio.to_thread(self._get_settings, user_id)

    async def set_setting(self, user_id: int, field: str, value: str) -> None:
        await asyncio.to_thread(self._set_setting, user_id, field, value)

    async def reset_settings(self, user_id: int) -> None:
        await asyncio.to_thread(self._reset_settings, user_id)

    async def get_mode(self, user_id: int) -> Optional[str]:
        return (await self.get_settings(user_id)).get("mode")

    async def set_mode(self, user_id: int, mode: str) -> None:
        await self.set_setting(user_id, "mode", mode)

    async def stats(self) -> dict:
        return await asyncio.to_thread(self._stats)

    async def export_csv(self, only_corrections: bool = False) -> bytes:
        return await asyncio.to_thread(self._export_csv, only_corrections)
