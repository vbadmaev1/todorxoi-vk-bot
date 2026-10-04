# -*- coding: utf-8 -*-
"""
clear_script.py — второй способ нарисовать тодо бичиг: семейство шрифтов
Clear Script (сам Clear Script, Biyir, Demberil, Garcaq, Zakaa).

ЧЕМ ОНИ ОТЛИЧАЮТСЯ ОТ ОСНОВНОГО

MongolianUniversalWhite работает с настоящим юникодом тодо бичиг: ему
подают ᡍᠠᠯᡅᡏᠠᡎ, а нужные формы букв выбирает HarfBuzz по таблицам GSUB.

Эти устроены иначе: в них нет ни одной монгольской кодовой точки, только
кириллица, латиница и цифры. Форму буквы задаёт не движок, а сам набор:
строчная и заглавная кириллические буквы — это разные начертания
(начальное, срединное, конечное), и «ХалимаЩ» рисуется как хальмг. То
есть шрифт ждёт особую запись, а не текст.

Запись у всех пяти одна и та же — различается только начертание,
поэтому правила общие, а выбор шрифта меняет лишь файл (FONT_FILES).
Начертания всё же расходятся настолько, что готовую запись приходится
подгонять под конкретный шрифт — этим занимается fit_to_font().

Эту запись строят правила из assets/clear_script_rules.json: статистика,
собранная выравниванием словаря (транслитерация ↔ запись для шрифта).
Здесь лежит только та половина, что нужна боту, — перевод транслитерации
в запись. Алгоритм: динамическое программирование по сегментации слова на
куски в 1–2 буквы, для каждого куска берётся самый частый вариант с
откатом по контексту (позиция в слове + соседние буквы → позиция →
общий).

ВАЖНО ПРО ТОЧНОСТЬ. Правила статистические: на словах из словаря они
воспроизводят запись почти всегда, на редких и незнакомых — могут
ошибиться в форме буквы. Это цена того, что шрифт не умеет выбирать формы
сам. Основной шрифт такой особенности не имеет, поэтому он и оставлен по
умолчанию.

ЗНАКИ ПРЕПИНАНИЯ. В присланных шрифтах их не было вовсе, поэтому бирга,
запятая, точка, четыре точки, тире и узкий неразрывный пробел дорисованы
в каждый отдельными глифами — см. tools/add_marks_to_fonts.py. Отступ
перед знаком заложен в сам глиф (слева от чернил), поэтому знак не
налезает на хвост предыдущей буквы и вставлять пробел в текст не нужно.
Здесь мы просто не выбрасываем знаки из текста, а переводим в те же коды,
что и основной путь, чтобы дальше сработала общая логика из
punctuation.py.

СТЕРЖЕНЬ И ПРОБЕЛЫ. Буквы соединяются перекрытием, и у части пар оно
выходит слишком сильным: предыдущая буква заливает петлю следующей
(тэйимю — хвост «м» закрывает «ю»). В шрифте для этого есть «_» —
короткий отрезок стержня; какой паре сколько его нужно, замерено по
каждому шрифту и лежит в assets/clear_script_gaps.json. Там же, в
fit_to_font(), пробел между словами удваивается: одинарный в этих
начертаниях слишком тесен.
"""

import json
import math
import re
import threading
import unicodedata
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ASSETS = _HERE.parent / "assets"
RULES_PATH = _ASSETS / "clear_script_rules.json"
GAPS_PATH = _ASSETS / "clear_script_gaps.json"

# Шрифты семейства: все ждут одну и ту же кириллическую запись, поэтому
# правила у них общие, а отличается только начертание. Ключ — то, что
# хранится в настройках пользователя.
FONT_FILES = {
    "clear": "clear_script.ttf",
    "biyir": "todo_biyir.ttf",
    "demberil": "todo_demberil.ttf",
    "garcaq": "todo_garcaq.ttf",
    "zakaa": "todo_zakaa.ttf",
}

# первый из них — для проверок и обратной совместимости
FONT_PATH = _ASSETS / FONT_FILES["clear"]


def font_path(key: str) -> Path:
    """Путь к файлу шрифта по ключу настроек."""
    return _ASSETS / FONT_FILES.get(key, FONT_FILES["clear"])

MAX_SPAN = 2        # сколько букв латиницы может лечь в один кусок
MIN_CTX_COUNT = 3   # ниже этого контекстное правило считаем ненадёжным

# долгота выносится отдельным символом: так её удобно выравнивать с
# мягким знаком, которым она записывается в шрифте
_MACRON_SPLIT = {
    "ā": "aˉ", "ē": "eˉ", "ī": "iˉ", "ō": "oˉ", "ū": "uˉ", "ȫ": "öˉ", "ǖ": "üˉ",
}

# Знаки препинания переводим в те же вертикальные формы, что и основной
# путь: дальше их подхватит punctuation.py и превратит в ᠂ ᠃ ᠅.
_PUNCT = {
    ",": "︐",
    ".": "︒",
    "!": "︕",
    "?": "︖",
}

# буквы транслитерации — всё остальное считаем «между словами»
_LETTERS_RE = re.compile(r"[a-zāēīōūȫǖöüγčšǰˉ]+", re.IGNORECASE)

# «_» в этих шрифтах — короткий отрезок стержня: удлиняет спину слова, не
# разрывая линию. Нужен там, где чернила следующей буквы уходят далеко
# влево и предыдущая заливает её петлю (тэйимю: хвост «м» закрывает «ю»).
# Какой паре сколько отрезков нужно — замерено по каждому шрифту отдельно,
# см. tools/fit_spine_gaps.py.
SPINE_MARK = "_"

# Слова в этих шрифтах стоят вплотную: обычного пробела мало, чтобы конец
# одного слова не читался началом следующего. Ставим два.
WORD_SPACE = "  "


def _load_gaps() -> dict:
    try:
        return json.loads(GAPS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_GAPS = None


def gaps_for(font_key: str) -> dict:
    """Таблица «пара букв -> сколько отрезков стержня» для этого шрифта."""
    global _GAPS
    if _GAPS is None:
        with _LOCK:
            if _GAPS is None:
                _GAPS = _load_gaps()
    return _GAPS.get(font_key, {})


def fit_to_font(record: str, font_key: str) -> str:
    """Подгоняет готовую запись под конкретный шрифт.

    Правила записи общие для всего семейства, а начертания разные: где у
    одного шрифта буквы сходятся нормально, у другого следующая буква
    попадает под хвост предыдущей. Поэтому сюда приезжает уже готовая
    запись, а здесь в неё доставляются отрезки стержня — по замеренной
    таблице — и разводятся слова.
    """
    table = gaps_for(font_key)
    out = []
    for i, ch in enumerate(record):
        if out and table:
            n = table.get(record[i - 1] + ch)
            if n:
                out.append(SPINE_MARK * n)
        out.append(WORD_SPACE if ch == " " else ch)
    return "".join(out)


def _split_lat(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = re.sub(r"\s+", "-", s)
    return "".join(_MACRON_SPLIT.get(c, c) for c in s)


class ClearScriptRules:
    """Транслитерация -> запись для шрифта Clear Script."""

    def __init__(self, path=None):
        data = json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))
        self.r_uni = data["r_uni"]
        self.r_pos = data["r_pos"]
        self.r_prev = data["r_prev"]
        self.r_next = data["r_next"]
        self.r_tri = data["r_tri"]
        self.unit_rate = data["unit_rate"]

    @staticmethod
    def _pick(table, key, min_count):
        d = table.get(key)
        return d if d and sum(d.values()) >= min_count else None

    def _variants(self, pos, prev, unit, nxt):
        """Откат по контексту: чем конкретнее правило, тем оно важнее."""
        return (self._pick(self.r_tri, f"{pos}|{prev}|{unit}|{nxt}", MIN_CTX_COUNT)
                or self._pick(self.r_prev, f"{pos}|{prev}|{unit}", MIN_CTX_COUNT)
                or self._pick(self.r_next, f"{pos}|{unit}|{nxt}", MIN_CTX_COUNT)
                or self._pick(self.r_pos, f"{pos}|{unit}", 1)
                or self._pick(self.r_uni, unit, 1))

    def _word(self, s: str) -> str:
        """Одно слово: ищем разбиение на куски с наибольшим суммарным весом."""
        n = len(s)
        best = [(-math.inf, None)] * (n + 1)
        best[0] = (0.0, None)
        for j in range(n):
            if best[j][0] == -math.inf:
                continue
            for k in range(1, MAX_SPAN + 1):
                if j + k > n:
                    break
                unit = s[j:j + k]
                if unit not in self.r_uni:
                    continue
                end = j + k == n
                pos = "S" if (j == 0 and end) else "I" if j == 0 else "F" if end else "M"
                prev = s[j - 1] if j > 0 else "#"
                nxt = s[j + k] if not end else "#"
                variants = self._variants(pos, prev, unit, nxt)
                if not variants:
                    continue
                chunk, count = max(variants.items(), key=lambda kv: kv[1])
                score = (best[j][0]
                         + math.log(self.unit_rate.get(unit, 1e-3) + 1e-9)
                         + math.log(count / sum(variants.values())))
                if score > best[j + k][0]:
                    best[j + k] = (score, (j, chunk))
        if best[n][1] is None:
            return s  # ничего не подобралось — отдаём как есть, видно будет сразу
        chunks, j = [], n
        while j > 0:
            prev_j, chunk = best[j][1]
            chunks.append(chunk)
            j = prev_j
        return "".join(reversed(chunks))

    def translit_to_font(self, translit: str) -> str:
        """Строка транслитерации -> строка, которую рисует Clear Script.

        Буквы идут через правила, всё остальное — знаки препинания, тире,
        пробелы — переносится один в один, только в те же коды, что и на
        основном пути. Поэтому текст остаётся текстом: и перенос по
        столбцам, и расстановка бирги работают как обычно.
        """
        from .translit_todo import _dashes_to_vertical

        out_lines = []
        for line in _dashes_to_vertical(str(translit)).split("\n"):
            out, pos = [], 0
            for m in _LETTERS_RE.finditer(line):
                out.append(self._between(line[pos:m.start()]))
                out.append(self._word(_split_lat(m.group())))
                pos = m.end()
            out.append(self._between(line[pos:]))
            out_lines.append("".join(out))
        return "\n".join(out_lines)

    @staticmethod
    def _between(chunk: str) -> str:
        """То, что между словами: пробелы, знаки, дефис-граница суффикса."""
        out = []
        for ch in chunk:
            if ch in _PUNCT:
                out.append(_PUNCT[ch])
            elif ch == "-":
                out.append("\u202f")      # граница суффикса, как в основном пути
            elif ch.isspace() or ch == "\uFE31":
                out.append(ch)
            # остальное (цифры, скобки) в этом шрифте всё равно не нужно
        return "".join(out)


_RULES = None
_LOCK = threading.Lock()


def get_rules() -> ClearScriptRules:
    """Ленивая загрузка: файл правил на четверть мегабайта."""
    global _RULES
    if _RULES is None:
        with _LOCK:
            if _RULES is None:
                _RULES = ClearScriptRules()
    return _RULES


def translit_to_font(translit: str) -> str:
    return get_rules().translit_to_font(translit)


def available() -> bool:
    return RULES_PATH.exists() and all(
        (_ASSETS / name).exists() for name in FONT_FILES.values()
    )


if __name__ == "__main__":
    for w in ["xalimaq", "ulus", "karou", "bolxu", "γazar-yēn", "kelen melen"]:
        print(f"  {w:14s} -> {translit_to_font(w)}")
