# -*- coding: utf-8 -*-
"""
pipeline.py — единая точка входа для бота: «дай текст и скажи, что нужно
на выходе». Здесь же замеряется время работы каждого шага.

Четыре сценария:
  1. TARGET_TRANSLIT — калмыцкая кириллица -> транслитерация на латинице
  2. TARGET_TODO     — кириллица / транслитерация -> тодо бичиг (юникод)
  3. TARGET_IMAGE    — кириллица / транслитерация / тодо бичиг -> картинка
  4. TARGET_FIX      — кириллица -> та же кириллица с исправленными
                       калмыцкими буквами (см. core/fix_letters.py)

Тип входного текста определяется автоматически (detect_script), так что
пользователю не нужно ничего указывать руками.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .translit_todo import (
    SCRIPT_CYRILLIC,
    SCRIPT_TODO,
    SCRIPT_TRANSLIT,
    SCRIPT_UNKNOWN,
    detect_script,
    todo_to_translit,
    translit_to_todo,
)
from .punctuation import DEFAULT_PUNCT, PUNCT_MODES, apply_punctuation

log = logging.getLogger(__name__)

TARGET_TRANSLIT = "translit"
TARGET_TODO = "todo"
TARGET_IMAGE = "image"
TARGET_FIX = "fix"

MAX_INPUT_CHARS = 10000

FONT_SIZES = {"small": 44, "medium": 64, "large": 96}
DEFAULT_FONT_SIZE = "medium"

# Два способа рисовать одно и то же — и шесть шрифтов на выбор.
#   universal — MongolianUniversalWhite: настоящий юникод тодо бичиг, формы
#               букв выбирает HarfBuzz;
#   остальные — семейство Clear Script: рисуют по особой кириллической
#               записи, которую строят статистические правила. Начертания
#               разные, правила общие. Подробности — в core/clear_script.py.
DEFAULT_FONT = "universal"


def _font_keys():
    from .clear_script import FONT_FILES

    return (DEFAULT_FONT,) + tuple(FONT_FILES)


FONTS = _font_keys()


@dataclass
class ImageOptions:
    """Как рисовать картинку. У каждого пользователя свои — хранятся в БД,
    меняются командой /settings."""

    fg: str = "black"
    bg: str = "white"
    size: str = DEFAULT_FONT_SIZE
    font: str = DEFAULT_FONT

    @property
    def font_size(self) -> int:
        return FONT_SIZES.get(self.size, FONT_SIZES[DEFAULT_FONT_SIZE])

    @property
    def max_column_height(self) -> int:
        # высота столбца соразмерна кеглю, иначе крупный шрифт ломает
        # перенос: на строку влезает слишком мало букв
        return int(os.environ.get("MAX_COLUMN_HEIGHT", "900")) * self.font_size // 64

    @property
    def uses_rules(self) -> bool:
        """Нужно ли строить кириллическую запись вместо юникода."""
        return self.font != DEFAULT_FONT

    @property
    def font_path(self):
        from .clear_script import font_path
        from .todo_image import DEFAULT_TODO_FONT

        if not self.uses_rules:
            return DEFAULT_TODO_FONT
        return str(font_path(self.font))

    def as_dict(self) -> dict:
        return {"fg": self.fg, "bg": self.bg, "size": self.size, "font": self.font}


class PipelineError(Exception):
    """Ошибка, текст которой можно показать пользователю как есть."""


@dataclass
class Result:
    target: str
    source_text: str
    source_script: str
    translit: Optional[str] = None
    todo: Optional[str] = None
    # кириллица из тодо бичиг или транслитерации (режим транслитерации),
    # см. core/to_cyrillic.py; None — не считали или не получилось
    cyrillic: Optional[str] = None
    # Листы картинки: [(io.BytesIO с PNG, (ширина, высота))]. Длинный текст
    # в одну картинку не влезает — см. render_todo_pages в todo_image.py.
    pages: list = field(default_factory=list)
    # столько листов пришлось отбросить, чтобы не заваливать чат
    pages_dropped: int = 0
    # False — картинка нарисована без шейпинга: буквы не соединены
    shaping_ok: bool = True
    # прозрачный фон: такую картинку надо слать документом, не фото
    transparent: bool = False
    # цвета текста и фона совпали, пришлось откатиться на чёрное по белому
    color_fallback: bool = False
    # Калмыцкие буквы (см. core/fix_letters.py). letters_flag — почему текст
    # похож на набранный без ә ө ү һ җ ң ('' — не похож); letter_fixes —
    # что исправлено, если пользователь включил исправление в /settings.
    letters_flag: str = ""
    letter_fixes: list = field(default_factory=list)
    # режим TARGET_FIX: исправленный текст
    fixed_text: Optional[str] = None
    elapsed_ms: float = 0.0
    steps_ms: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def image(self):
        """Первый лист. Для кода, которому хватает одной картинки."""
        return self.pages[0][0] if self.pages else None

    @property
    def image_size(self):
        return self.pages[0][1] if self.pages else None

    @property
    def text_output(self) -> str:
        """Главный текстовый результат — то, что пойдёт в БД и в ответ."""
        if self.target == TARGET_TRANSLIT:
            return self.translit or ""
        if self.target == TARGET_FIX:
            return self.fixed_text or ""
        return self.todo or ""


def _check_input(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise PipelineError("Пустой текст — пришлите слово или предложение.")
    if len(text) > MAX_INPUT_CHARS:
        raise PipelineError(
            f"Слишком длинный текст: {len(text)} символов, "
            f"максимум {MAX_INPUT_CHARS}."
        )
    return text


def _to_translit(text: str, script: str, res: Result, fix_letters: bool = False) -> str:
    """Приводит любой вход к транслитерации проекта."""
    if script == SCRIPT_TRANSLIT:
        return text
    if script == SCRIPT_TODO:
        t0 = time.perf_counter()
        out = todo_to_translit(text)
        res.steps_ms["todo→translit"] = (time.perf_counter() - t0) * 1000
        return out
    if script == SCRIPT_CYRILLIC:
        from .transliterate import transliterate_with_stats

        text = _check_letters(text, res, fix_letters)
        t0 = time.perf_counter()
        # Кириллицу приводим к строчным. Заглавная буква сама по себе
        # безобидна, но восстановление регистра после модели ломает
        # транслитерацию: «Җирһл» давало J̌irγal — J с отдельным
        # диакритическим знаком, которого нет в таблицах, и дальше запись
        # для шрифта получалась другой, чем у «җирһл». Тодо бичиг
        # прописных букв не знает, так что терять тут нечего.
        out, stats = transliterate_with_stats(text.lower())
        res.steps_ms["модель"] = (time.perf_counter() - t0) * 1000
        res.stats.update(stats)
        return out
    raise PipelineError(
        "Не удалось распознать текст. Пришлите калмыцкую кириллицу, "
        "транслитерацию на латинице или тодо бичик."
    )


def _to_cyrillic(translit: str, res: Result) -> Optional[str]:
    t0 = time.perf_counter()
    try:
        from .to_cyrillic import to_cyrillic

        out = to_cyrillic(translit)
    except Exception:
        # без моделей кириллицы режим работает как раньше
        log.exception("перевод в кириллицу не удался")
        out = None
    res.steps_ms["кириллица"] = (time.perf_counter() - t0) * 1000
    return out


def _check_letters(text: str, res: Result, fix: bool) -> str:
    """Кириллица без калмыцких букв: исправить (если пользователь включил)
    или только заметить — тогда бот подскажет, что есть такая настройка."""
    from . import fix_letters

    t0 = time.perf_counter()
    try:
        if fix:
            fixed = fix_letters.fix_text(text)
            res.letters_flag = fixed.detection.flag
            res.letter_fixes = fixed.fixes
            text = fixed.text
        else:
            res.letters_flag = fix_letters.detect(text).flag
    except Exception:
        # без словаря бот работает как раньше: переводит текст как есть
        log.exception("проверка калмыцких букв не удалась")
    res.steps_ms["буквы"] = (time.perf_counter() - t0) * 1000
    return text


def process(
    text: str,
    target: str,
    options: Optional[ImageOptions] = None,
    fix_letters: bool = False,
    punctuation: str = DEFAULT_PUNCT,
) -> Result:
    """Основная функция. Синхронная и не быстрая (модель) — в боте её
    нужно звать через asyncio.to_thread.

    options — настройки картинки конкретного пользователя; для режимов
    транслитерации и тодо бичиг не используются. fix_letters — исправлять
    ли кириллицу, набранную без ә ө ү һ җ ң (настройка в /settings,
    по умолчанию выключена). punctuation — знаки препинания тодо бичиг
    (PUNCT_* из punctuation.py, по умолчанию без знаков)."""
    opts = options or ImageOptions()
    text = _check_input(text)
    script = detect_script(text)
    if script == SCRIPT_UNKNOWN:
        raise PipelineError(
            "Не удалось распознать текст. Пришлите калмыцкую кириллицу, "
            "транслитерацию на латинице или тодо бичик."
        )

    res = Result(target=target, source_text=text, source_script=script)
    started = time.perf_counter()

    if target == TARGET_TRANSLIT:
        res.translit = _to_translit(text, script, res, fix_letters)
        if script in (SCRIPT_TODO, SCRIPT_TRANSLIT):
            # обратно в кириллицу: для тодо бичиг — вместе с транслитерацией,
            # для латиницы только это и есть результат
            res.cyrillic = _to_cyrillic(res.translit, res)

    elif target == TARGET_FIX:
        if script != SCRIPT_CYRILLIC:
            raise PipelineError(
                "Исправление работает с калмыцкой кириллицей: пришлите текст "
                "кириллицей, например «Сян бяянт!»."
            )
        from . import fix_letters

        t0 = time.perf_counter()
        # человек сам попросил проверить текст — правим незнакомые слова,
        # даже если текст в целом набран правильно
        fixed = fix_letters.fix_text(text, force=True)
        res.fixed_text = fixed.text
        res.letters_flag = fixed.detection.flag
        res.letter_fixes = fixed.fixes
        res.steps_ms["буквы"] = (time.perf_counter() - t0) * 1000

    elif target == TARGET_TODO:
        if script == SCRIPT_TODO:
            raise PipelineError(
                "Этот текст уже записан тодо бичик. Если нужна картинка — "
                "используйте /image."
            )
        res.translit = _to_translit(text, script, res, fix_letters)
        t0 = time.perf_counter()
        res.todo = apply_punctuation(translit_to_todo(res.translit), punctuation)
        res.steps_ms["translit→тодо"] = (time.perf_counter() - t0) * 1000

    elif target == TARGET_IMAGE:
        from .todo_image import (
            SHAPING_OK,
            ShapingUnavailable,
            render_todo_pages,
            require_shaping,
        )

        try:
            require_shaping()
        except ShapingUnavailable as exc:
            # сюда попадаем только при STRICT_SHAPING=1
            raise PipelineError(
                "Рендер картинок отключён: окружение не умеет соединять "
                "буквы тодо бичик (STRICT_SHAPING=1). Подробности — в логах."
            ) from exc
        # без движка раскладки картинку всё равно рисуем, но честно помечаем
        # результат, чтобы никто не принял несоединённые буквы за письмо
        res.shaping_ok = SHAPING_OK

        if script == SCRIPT_TODO:
            res.todo = text
        else:
            res.translit = _to_translit(text, script, res, fix_letters)
            t0 = time.perf_counter()
            res.todo = translit_to_todo(res.translit)
            res.steps_ms["translit→тодо"] = (time.perf_counter() - t0) * 1000
        # Что уедет в шрифт, зависит от выбранного:
        #   universal — юникод тодо бичиг;
        #   остальные — особая кириллическая запись по правилам.
        # Знаки препинания в обоих случаях расставляются одинаково: в
        # семейство Clear Script они дорисованы отдельными глифами, см.
        # tools/add_marks_to_fonts.py.
        t0 = time.perf_counter()
        plain_todo = res.todo
        res.todo = apply_punctuation(plain_todo, punctuation)
        if opts.uses_rules:
            from .clear_script import fit_to_font, translit_to_font

            source = res.translit or todo_to_translit(plain_todo)
            # Знаки — до подгонки: она удваивает пробелы между словами, а
            # убранный знак оставляет после себя пробел, который надо
            # схлопнуть с соседним. Запись общая для семейства, подгонка
            # под начертание — своя.
            record = apply_punctuation(translit_to_font(source), punctuation)
            render_text = fit_to_font(record, opts.font)
        else:
            render_text = res.todo
        res.steps_ms["знаки"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            res.pages, meta = render_todo_pages(
                render_text,
                font_size=opts.font_size,
                max_column_height=opts.max_column_height,
                fg=opts.fg,
                bg=opts.bg,
                font_path=opts.font_path,
            )
        except ValueError as exc:
            # после снятия знаков препинания не осталось ни одной буквы
            raise PipelineError(
                "Нечего нарисовать: в тексте нет слов, одни знаки препинания."
            ) from exc
        res.transparent = meta["transparent"]
        res.color_fallback = meta["color_fallback"]
        res.pages_dropped = meta["truncated"]
        res.steps_ms["рендер"] = (time.perf_counter() - t0) * 1000

    else:
        raise PipelineError(f"Неизвестная операция: {target}")

    res.elapsed_ms = (time.perf_counter() - started) * 1000
    return res


SCRIPT_TITLES = {
    SCRIPT_CYRILLIC: "калмыцкая кириллица",
    SCRIPT_TRANSLIT: "транслитерация (латиница)",
    SCRIPT_TODO: "тодо бичиг",
    SCRIPT_UNKNOWN: "не определено",
}
