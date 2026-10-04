# -*- coding: utf-8 -*-
"""
to_cyrillic.py — транслитерация (например, распознанного фото) в
современную калмыцкую кириллицу.

Трудность в том, что слова не совпадают: одно кириллическое слово в тодо
бичиг — часто несколько слов (келәр — kele bēr, орнд — oron du, үгиг —
üge igi или ügeyigi). Поэтому два шага:

  1. core/segmenter.py решает для каждого пробела, склеивать ли соседние
     куски в одно кириллическое слово;
  2. модель translit→кириллица (тот же seq2seq, что core/model_numpy.py,
     веса model/translit2cyr_model.npz) переводит окна из нескольких
     таких слов. Она обучена на окнах в 1–4 слова: соседние слова дают
     контекст, а резать окна можно только по границам слов из шага 1 —
     иначе kele и bēr попали бы в разные окна и «келәр» не собрался бы.

Обе модели и данные — todorxoi-inference/translit2cyr. На val: сегментатор
верно решает 99.8 % пробелов, вся цепочка — 94 % слов (69 % строк без
единой ошибки), ~14 мс на строку в 7 слов.

Регистра в тодо бичиг нет: заглавной делаем первую букву текста и букву
после . ! ?
"""

import os
import re
from pathlib import Path
from typing import List

from . import segmenter
from .model_numpy import load

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = _HERE.parent / "model" / "translit2cyr_model.npz"

# Окно — до стольких кириллических слов подряд и не длиннее MAX_WINDOW_CHARS
# символов (обучение — окна в 1–4 слова до 70 символов). Замер на 1500
# строках val, всё через сегментатор: окно 1 — 93.9 % слов, 2 — 94.1 %,
# 3 — 93.9 %, 4 — 93.7 %. Словарь пар из .npz вместо модели хуже (90.7 %):
# в нём много шумных источников, поэтому он не используется.
MAX_WORDS = 2
MAX_WINDOW_CHARS = 60

_SPACE_BEFORE_PUNCT = re.compile(r" +([,.!?])")
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)(\w)")


def model_path() -> str:
    return os.environ.get("TRANSLIT2CYR_MODEL_PATH") or str(_DEFAULT_MODEL)


def get_model():
    return load(model_path())


def warmup() -> dict:
    t = to_cyrillic("kele bēr")
    return {"segmenter": segmenter.model_path(), "model": model_path(), "sample": t}


def windows(groups: List[List[str]]) -> List[str]:
    """Группы кусков -> окна для модели. Знак препинания — своя группа, он
    прилипает к окну слева и в число слов не входит."""
    out, cur, n, chars = [], [], 0, 0
    for g in groups:
        s = " ".join(g)
        word = s[:1].isalpha()
        if word and n and (n >= MAX_WORDS or chars + len(s) + 1 > MAX_WINDOW_CHARS):
            out.append(" ".join(cur))
            cur, n, chars = [], 0, 0
        cur.append(s)
        n += word
        chars += len(s) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def _tidy(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _capitalize(text: str) -> str:
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def lines_to_cyrillic(lines: List[str]) -> List[str]:
    """Строки транслитерации -> строки кириллицы, одна к одной (столбцы OCR).
    Регистр не трогаем: его восстанавливает to_cyrillic по всему тексту."""
    seg = segmenter.get_model()
    model = get_model()
    out = []
    for line in lines:
        parts = [model.translate_word(w) for w in windows(seg.segment(line))]
        out.append(_tidy(" ".join(p for p in parts if p)))
    return out



def to_cyrillic(text: str) -> str:
    """Транслитерация (любой текст, по строкам) -> кириллица."""
    return _capitalize("\n".join(lines_to_cyrillic(text.split("\n"))))
