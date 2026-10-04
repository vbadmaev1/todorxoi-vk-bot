# -*- coding: utf-8 -*-
"""
restore_from_jsonl.py — собрать базу заново из append-only лога.

Ради этого лог и пишется параллельно с SQLite: файл базы можно потерять
при переезде, пересоздании контейнера или очистке диска, а events.jsonl
это обычный текст, который несложно скопировать даже из веб-терминала
хостинга (`cat /app/data/events.jsonl`) и сохранить куда угодно.

Использование:

    python -m tools.restore_from_jsonl events.jsonl
    python -m tools.restore_from_jsonl events.jsonl --db data/todorxoi.sqlite3

Скрипт не трогает существующие строки: записи вставляются по их
собственным id, повторный прогон того же лога ничего не задвоит. Значит,
восстановление можно повторять и можно склеивать несколько логов,
накопившихся в разное время.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.storage import _SCHEMA, Storage  # noqa: E402

REQUEST_FIELDS = [
    "created_at", "user_id", "username", "chat_id", "target", "source_script",
    "input_text", "translit", "todo", "cyrillic", "elapsed_ms", "steps_json", "ok",
    "error", "letters_json",
]
FEEDBACK_FIELDS = [
    "request_id", "created_at", "user_id", "username", "rating", "correction",
]


def restore(jsonl_path: Path, db_path: Path) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    # база могла быть заведена старой версией — без новых колонок
    for sql in Storage._MIGRATIONS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass

    counts = {"requests": 0, "feedback": 0, "skipped": 0, "broken": 0}

    with jsonl_path.open(encoding="utf-8") as fh:
        for num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # обрезанная последняя строка — обычное дело, если лог
                # копировали на ходу; молча пропускаем, но считаем
                counts["broken"] += 1
                print(f"  строка {num}: не разобралась, пропускаю")
                continue

            kind = row.get("event")
            if kind == "request":
                values = [row.get("id")] + [row.get(f) for f in REQUEST_FIELDS]
                cur = conn.execute(
                    "INSERT OR IGNORE INTO requests (id, "
                    + ", ".join(REQUEST_FIELDS)
                    + ") VALUES (" + ", ".join("?" * (len(REQUEST_FIELDS) + 1)) + ")",
                    values,
                )
                counts["requests" if cur.rowcount else "skipped"] += 1
            elif kind == "feedback":
                cur = conn.execute(
                    "INSERT OR IGNORE INTO feedback ("
                    + ", ".join(FEEDBACK_FIELDS)
                    + ") VALUES (" + ", ".join("?" * len(FEEDBACK_FIELDS)) + ")",
                    [row.get(f) for f in FEEDBACK_FIELDS],
                )
                counts["feedback" if cur.rowcount else "skipped"] += 1
            else:
                counts["broken"] += 1

    conn.commit()
    conn.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path, help="файл events.jsonl")
    ap.add_argument(
        "--db", type=Path, default=Path("data/todorxoi.sqlite3"),
        help="куда писать базу (по умолчанию data/todorxoi.sqlite3)",
    )
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"Нет такого файла: {args.jsonl}")

    print(f"Читаю {args.jsonl} -> {args.db}")
    counts = restore(args.jsonl, args.db)
    print(
        f"\nГотово. Запросов добавлено: {counts['requests']}, "
        f"оценок: {counts['feedback']}, "
        f"уже было в базе: {counts['skipped']}, "
        f"нечитаемых строк: {counts['broken']}"
    )


if __name__ == "__main__":
    main()
