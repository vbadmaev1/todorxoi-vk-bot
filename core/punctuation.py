# -*- coding: utf-8 -*-
"""
punctuation.py — знаки препинания тодо бичиг.

Ставить ли их, решает пользователь в /settings; настройка действует на
всё, где бот выдаёт тодо бичиг: текст (/todo), картинку (/image) и
распознанное фото. Три режима:

  off   — без знаков (по умолчанию): все знаки препинания убираются,
          остаются только слова;
  frame — только разметка всего текста: бирга в начале и четыре точки
          в конце, знаки внутри убираются;
  all   — все знаки: бирга, запятые, точки, вопрос и восклицание, четыре
          точки в конце.

Знаки:

  ᠀  бирга (U+1800)      — в самом начале текста, знак начала записи;
  ᠂  запятая (U+1802)    — вместо обычной;
  ᠃  точка (U+1803)      — конец предложения внутри текста;
  ᠅  четыре точки (U+1805) — конец всего текста.

Точка в самом конце заменяется на четыре точки. Если текст кончается
вопросительным или восклицательным знаком, знак остаётся, а четыре точки
добавляются после него.

На вход приходит уже готовый юникод тодо бичиг, где знаки препинания
записаны вертикальными презентационными формами (︒ ︐ ︕ ︖) — так их
превращает таблица в translit_todo.py, — или настоящими монгольскими
(᠂ ᠃), как их читает модель распознавания фото.
"""

import re

BIRGA = "᠀"
COMMA = "᠂"
FULL_STOP = "᠃"
FOUR_DOTS = "᠅"

PUNCT_OFF = "off"
PUNCT_FRAME = "frame"
PUNCT_ALL = "all"
PUNCT_MODES = (PUNCT_OFF, PUNCT_FRAME, PUNCT_ALL)
DEFAULT_PUNCT = PUNCT_OFF

# то, что приезжает из таблицы translit_todo.py
_SRC_COMMA = "︐"
_SRC_STOP = "︒"
_SRC_EXCL = "︕"
_SRC_QUES = "︖"

_TERMINAL = (_SRC_EXCL, _SRC_QUES)
_STOPS = (_SRC_STOP, FULL_STOP)

# Всё, что убирается в режимах off и frame: вертикальные формы из таблицы
# (︐ ︑ ︒ ︕ ︖ ︽ ︾ и тире ︱), монгольские знаки — и то, что таблица
# пропускает как есть: точка с запятой, скобки, кавычки-ёлочки, многоточие.
_MARKS = "︐︑︒︕︖︽︾︱᠀᠁᠂᠃᠄᠅᠈᠉;()[]«»“”„…"
_MARKS_RE = re.compile(f"[{re.escape(_MARKS)}]+")
_SPACES_RE = re.compile(" {2,}")


def strip_marks(text: str) -> str:
    """Убирает знаки препинания, не трогая слова и переводы строк.

    Знак меняется на пробел, а не выбрасывается: в «ulus,mani» без пробела
    слова иначе слиплись бы в одно."""
    lines = []
    for line in text.split("\n"):
        line = _SPACES_RE.sub(" ", _MARKS_RE.sub(" ", line))
        lines.append(line.strip(" "))
    return "\n".join(lines)


def _frame(text: str) -> str:
    """Бирга перед первой непустой строкой, четыре точки — в конце
    последней. Точка в самом конце — это конец всего текста, её заменяют
    четыре точки; вопрос и восклицание остаются, четыре точки встают
    после них."""
    lines = text.split("\n")
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if not filled:
        return text
    last = lines[filled[-1]].rstrip()
    if last[-1] in _STOPS:
        last = last[:-1]
    lines[filled[-1]] = last + FOUR_DOTS
    lines[filled[0]] = BIRGA + lines[filled[0]].lstrip(" ")
    return "\n".join(lines)


def add_punctuation(todo_text: str, frame: bool = True) -> str:
    """Расставляет все знаки препинания тодо бичиг в готовой строке.

    frame=False — без бирги в начале и без четырёх точек в конце: точка
    в конце тогда остаётся обычной."""
    if not todo_text:
        return todo_text
    text = todo_text.replace(_SRC_COMMA, COMMA)
    if frame:
        text = _frame(text)
    return text.replace(_SRC_STOP, FULL_STOP)


def apply_punctuation(todo_text: str, mode: str = DEFAULT_PUNCT) -> str:
    """Знаки препинания по настройке пользователя (PUNCT_*)."""
    if not todo_text:
        return todo_text
    if mode == PUNCT_ALL:
        # тире в начале реплики приходит с пробелом перед собой: « ︱ сән»
        lines = [line.strip(" ") for line in todo_text.split("\n")]
        return add_punctuation("\n".join(lines))
    text = strip_marks(todo_text)
    return _frame(text) if mode == PUNCT_FRAME else text
