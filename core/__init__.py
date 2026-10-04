# -*- coding: utf-8 -*-
"""Ядро проекта: транслитерация, конвертация в тодо бичиг и рендер картинки.

Тяжёлые импорты (torch, PIL) намеренно НЕ делаются здесь — они подтягиваются
лениво внутри pipeline.process(), чтобы бот стартовал быстро.
"""

from .pipeline import (  # noqa: F401
    DEFAULT_FONT,
    DEFAULT_FONT_SIZE,
    DEFAULT_PUNCT,
    FONT_SIZES,
    FONTS,
    MAX_INPUT_CHARS,
    PUNCT_MODES,
    SCRIPT_TITLES,
    TARGET_FIX,
    TARGET_IMAGE,
    TARGET_TODO,
    TARGET_TRANSLIT,
    ImageOptions,
    PipelineError,
    Result,
    process,
)
from .translit_todo import (  # noqa: F401
    detect_script,
    todo_to_translit,
    translit_to_todo,
)

__all__ = [
    "process",
    "Result",
    "ImageOptions",
    "FONT_SIZES",
    "FONTS",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_FONT",
    "PUNCT_MODES",
    "DEFAULT_PUNCT",
    "PipelineError",
    "TARGET_TRANSLIT",
    "TARGET_TODO",
    "TARGET_IMAGE",
    "TARGET_FIX",
    "SCRIPT_TITLES",
    "MAX_INPUT_CHARS",
    "detect_script",
    "translit_to_todo",
    "todo_to_translit",
]
