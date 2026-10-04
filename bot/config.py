# -*- coding: utf-8 -*-
"""Конфигурация бота. Всё читается из переменных окружения (.env)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv не обязателен: на хостинге переменные обычно задаются иначе
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "да"}


def _int_list(name: str) -> list:
    raw = os.environ.get(name, "")
    out = []
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            try:
                out.append(int(piece))
            except ValueError:
                pass
    return out


@dataclass
class Config:
    # ключ доступа сообщества: Управление → Работа с API → Ключи доступа
    vk_token: str = ""
    # id сообщества; пусто — бот спросит его у VK по ключу
    group_id: int = 0
    # включить Long Poll и нужные события в настройках сообщества на старте
    # (ключу нужно право «управление сообществом»; без него — инструкция в лог)
    setup_longpoll: bool = True
    data_dir: Path = BASE_DIR / "data"
    db_path: Path = BASE_DIR / "data" / "todorxoi.sqlite3"
    jsonl_path: Path = BASE_DIR / "data" / "events.jsonl"
    model_path: Path = BASE_DIR / "model" / "translit_model.npz"
    font_path: Path = BASE_DIR / "assets" / "MongolianUniversalWhite.ttf"
    ocr_model_path: Path = BASE_DIR / "model" / "todo_ocr_int8.onnx"
    # в ответ на фото ещё и картинка с рамками столбцов (~50 мс и один JPEG)
    ocr_overlay: bool = True
    admin_ids: list = field(default_factory=list)
    log_level: str = "INFO"
    warmup: bool = True
    # показывать кнопки 👍/👎 ещё и под обычной транслитерацией,
    # а не только под тодо бичиг и картинкой
    feedback_on_translit: bool = True
    # ссылка на новости под /start и /help; пусто — без неё
    news_url: str = "https://t.me/todorxoi_uzuq"
    # сколько сообщений обрабатывать одновременно
    workers: int = 8

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            vk_token=os.environ.get("VK_TOKEN", "").strip(),
            group_id=abs(int(os.environ.get("VK_GROUP_ID", "0") or 0)),
            setup_longpoll=_bool("VK_SETUP_LONGPOLL", True),
            admin_ids=_int_list("ADMIN_IDS"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            warmup=_bool("WARMUP", True),
            feedback_on_translit=_bool("FEEDBACK_ON_TRANSLIT", True),
            ocr_overlay=_bool("OCR_OVERLAY", True),
            news_url=os.environ.get("NEWS_URL", cls.news_url).strip(),
            workers=max(1, int(os.environ.get("WORKERS", "8"))),
        )
        if os.environ.get("DATA_DIR"):
            cfg.data_dir = Path(os.environ["DATA_DIR"]).expanduser().resolve()
            cfg.db_path = cfg.data_dir / "todorxoi.sqlite3"
            cfg.jsonl_path = cfg.data_dir / "events.jsonl"
        if os.environ.get("DB_PATH"):
            cfg.db_path = Path(os.environ["DB_PATH"]).expanduser().resolve()
        if os.environ.get("JSONL_PATH"):
            cfg.jsonl_path = Path(os.environ["JSONL_PATH"]).expanduser().resolve()
        if os.environ.get("MODEL_PATH"):
            cfg.model_path = Path(os.environ["MODEL_PATH"]).expanduser().resolve()
        if os.environ.get("FONT_PATH"):
            cfg.font_path = Path(os.environ["FONT_PATH"]).expanduser().resolve()
        if os.environ.get("OCR_MODEL_PATH"):
            cfg.ocr_model_path = Path(os.environ["OCR_MODEL_PATH"]).expanduser().resolve()

        # ядро читает эти пути из окружения — синхронизируем обратно,
        # чтобы core/ и bot/ точно смотрели в одно и то же место
        os.environ["MODEL_PATH"] = str(cfg.model_path)
        os.environ["FONT_PATH"] = str(cfg.font_path)
        os.environ["OCR_MODEL_PATH"] = str(cfg.ocr_model_path)

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        return cfg

    def validate(self) -> None:
        if not self.vk_token:
            raise RuntimeError(
                "Не задан VK_TOKEN. Скопируйте .env.example в .env и впишите "
                "ключ доступа сообщества (Управление → Работа с API)."
            )
        if not self.model_path.exists():
            raise RuntimeError(f"Не найден чекпойнт модели: {self.model_path}")
        if not self.font_path.exists():
            raise RuntimeError(f"Не найден шрифт тодо бичиг: {self.font_path}")

    def is_admin(self, user_id: int) -> bool:
        # пока ADMIN_IDS не заполнен, админских команд нет ни у кого —
        # так выгрузку фидбэка не сможет забрать случайный человек
        return user_id in self.admin_ids
