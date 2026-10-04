# -*- coding: utf-8 -*-
"""
fix_letters.py — калмыцкий текст, набранный без калмыцких букв.

Люди часто пишут «бяядл» вместо «бәәдл», «hалун» вместо «һалун», «кюн»
вместо «күн»: на клавиатуре нет ә ө ү һ җ ң, и их заменяют русскими или
латинскими буквами. Модель транслитерации ждёт правильную орфографию и на
таком вводе ошибается в трёх словах из четырёх. Здесь такой текст
узнаётся и, если пользователь включил это в /settings, исправляется.

Как устроено:
  1. Буквы-двойники (латинская h, ә из азербайджанской раскладки, ҥ из
     якутской…) заменяются калмыцкими — там, где они стоят внутри
     кириллического слова.
  2. Текст целиком проверяется: действительно ли он написан без спецбукв.
     Русский текст отсекается сразу. Дальше — триггеры, от жёстких («ё»,
     «яя», согласная + я в незнакомом слове) до словарных. Правильный
     текст, где спецбуквы на месте, дальше не проверяется вовсе.
  3. В помеченном тексте каждое НЕЗНАКОМОЕ слово ищется в словаре: какие
     правильные слова могли дать такую запись. Кандидаты ранжируются по
     частоте и по тому, насколько правдоподобны нужные замены (ә -> я
     обычное дело, җ -> дж редкость), с поправкой на привычки автора в
     этом же тексте.
  4. Слово из словаря не меняется никогда: и «цаган», и «цаһан», и «дакад»,
     и «дәкәд» — настоящие слова, по буквам их не различить.
  5. Если целого слова в словаре нет, слово делится на основу и окончание:
     основа ищется в словаре, окончание — среди окончаний, собранных из
     словаря, причём гласные окончания согласуются с рядом основы
     (дугарна -> дуһар + на -> дуһарна; не «нә»: основа твёрдая).
  6. Если не нашлось и так, слово остаётся как есть — дальше его разберёт
     модель. Исправлять «с одной опечаткой» пробовали: незнакомое слово
     почти всегда просто отсутствует в словаре, а не опечатано, и такая
     правка была верной в 1% случаев.

Словарь с частотами — model/letters_vocab.tsv.gz, собирается скриптом
tools/build_letters_vocab.py из корпуса и словаря модели. Замеры и
примеры — research/no_special_letters/.
"""

import gzip
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
DEFAULT_VOCAB = _HERE.parent / "model" / "letters_vocab.tsv.gz"

SPECIAL = set("әөүһҗң")
CONS = set("бвгджзйклмнпрстфхцчшщһҗң")

# --------------------------------------------------------------------------
# Двойники
# --------------------------------------------------------------------------
# Выглядят как калмыцкие буквы (или как их привычная замена), приходят с
# чужих раскладок. Пользователь их не отличает — мы обязаны.
LOOKALIKE = {
    "h": "һ", "H": "Һ", "ҳ": "һ", "Ҳ": "Һ", "ғ": "һ", "Ғ": "Һ",
    "ə": "ә", "Ə": "Ә", "ǝ": "ә", "ӓ": "ә", "Ӓ": "Ә", "ä": "ә", "Ä": "Ә",
    "ӧ": "ө", "Ӧ": "Ө", "ö": "ө", "Ö": "Ө", "ɵ": "ө", "Ɵ": "Ө", "ѳ": "ө", "Ѳ": "Ө",
    "ӱ": "ү", "Ӱ": "Ү", "ü": "ү", "Ü": "Ү", "ұ": "ү", "Ұ": "Ү", "Y": "Ү",
    "ҷ": "җ", "Ҷ": "Җ", "ӂ": "җ", "Ӂ": "Җ", "ӝ": "җ", "Ӝ": "Җ",
    "ҥ": "ң", "Ҥ": "Ң", "ӈ": "ң", "Ӈ": "Ң", "ӊ": "ң", "Ӊ": "Ң", "ŋ": "ң", "Ŋ": "Ң",
}
# латиница, неотличимая от кириллицы на глаз
LATIN_TWINS = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "k": "к", "m": "м", "t": "т", "b": "в",
    "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "X": "Х", "K": "К",
    "M": "М", "T": "Т", "B": "В",
}
_CYR = "а-яёА-ЯЁәөүһҗңӘӨҮҺҖҢ"
_HAS_CYR = re.compile(f"[{_CYR}]")
_FOREIGN = re.compile("[a-zA-Z" + "".join(LOOKALIKE) + "]")
_ANY_TOKEN = re.compile(r"[\w'-]+")


def normalize_lookalikes(text: str) -> str:
    """Двойники внутри кириллических слов -> калмыцкие буквы. Чисто
    латинские слова не трогаем: это транслитерация, а не опечатка."""
    text = unicodedata.normalize("NFC", text)

    def fix(m):
        tok = m.group(0)
        if not _HAS_CYR.search(tok) or not _FOREIGN.search(tok):
            return tok
        return "".join(LOOKALIKE.get(c) or LATIN_TWINS.get(c) or c for c in tok)

    return _ANY_TOKEN.sub(fix, text)


# --------------------------------------------------------------------------
# Какие замены бывают и насколько они обычны
# --------------------------------------------------------------------------
# Цена замены: чем больше, тем реже люди так пишут. Не вероятности в
# строгом смысле, а штраф в тех же единицах, что и log(частоты) слова.
# «Родная» буква стоит 0: человек мог и набрать её.
SUB_COST = {
    "ә": {"я": 0.3, "а": 0.5, "э": 0.8, "е": 1.0},
    "ө": {"о": 0.3, "ё": 0.5, "е": 2.0, "э": 2.0},
    "ү": {"у": 0.3, "ю": 0.5},
    "һ": {"г": 0.5, "х": 1.0, "": 2.5},
    "җ": {"ж": 0.3, "дж": 1.0},
    "ң": {"н": 0.3, "нг": 1.0},
}
# во сколько раз дешевле замена, которую автор уже делал в этом тексте
HABIT_DISCOUNT = 0.3
# вес штрафа относительно log(частоты)
COST_WEIGHT = 2.0
# Разбор «основа + окончание»: окончание до 5 букв, основа от 3 букв и
# встречалась не реже 3 раз, окончание встречалось после основ того же
# ряда не реже 30 раз. Штраф за деление — чтобы целое словарное слово
# всегда выигрывало у разобранного.
SUFFIX_MAX = 5
STEM_MIN = 3
STEM_MIN_FREQ = 3
SUFFIX_MIN_COUNT = 30
SPLIT_PENALTY = 1.0


def skeleton(w: str) -> str:
    """Ключ индекса: одинаков у правильного слова и у любой его порчи.
    «дж» и «нг» не склеиваем — это бывают и настоящие д+ж, н+г (наадҗ ->
    наадж); замены җ -> дж, ң -> нг ищутся отдельными запросами."""
    w = w.replace("җ", "ж").replace("ң", "н")
    w = re.sub("[әяаэеөёо]", "a", w)   # е/э бывают и ә, и ө — один класс
    w = re.sub("[үюу]", "u", w)
    w = re.sub("[һгх]", "g", w)
    return w


def _query_keys(w: str) -> Set[str]:
    keys = {w}
    if "дж" in w:
        keys.add(w.replace("дж", "ж"))
    if "нг" in w:
        keys |= {k.replace("нг", "н") for k in list(keys)}
    return {skeleton(k) for k in keys}


def vowel_row(w: str) -> str:
    """Ряд слова по гласным: 'b' — задний (а о у), 'f' — передний (ә ө ү е
    э), 'n' — только и, 'm' — смешанный (заимствования). От ряда основы
    зависят гласные окончания и выбор г/һ, к/х: перед а, у почти всегда
    пишется һ и х, перед е — только г и к."""
    body = w.replace("я", "а").replace("ю", "у").replace("ё", "о")
    back, front = bool(set(body) & set("аоуы")), bool(set(body) & set("әөүеэ"))
    return "m" if back and front else "b" if back else "f" if front else "n"


def channel(v: str, w: str) -> Optional[Tuple[float, Tuple]]:
    """Можно ли из правильного v получить запись w. Возвращает (цена,
    использованные замены) самого дешёвого пути или None."""

    @lru_cache(None)
    def go(i: int, j: int):
        if i == len(v):
            return (0.0, ()) if j == len(w) else None
        best = None

        def take(res, add, sub):
            nonlocal best
            if res is not None:
                c = res[0] + add
                if best is None or c < best[0]:
                    best = (c, res[1] + ((sub,) if sub else ()))

        ch = v[i]
        if j < len(w) and w[j] == ch:
            take(go(i + 1, j + 1), 0.0, None)
        for rep, cost in SUB_COST.get(ch, {}).items():
            if w.startswith(rep, j):
                take(go(i + 1, j + len(rep)), cost, (ch, rep))
        return best

    return go(0, 0)


# --------------------------------------------------------------------------
# Русский текст
# --------------------------------------------------------------------------
# Без этого фильтра почти любой русский текст выглядит как «калмыцкий без
# спецбукв»: длинный, без ә и ү, с «ё». Два признака: служебные слова и
# слова на гласную (калмыцкая орфография конечные гласные почти всегда
# опускает: хальмг, келн, улс). У калмыцкого 0.5% и 10%, у русского ~24%
# и ~60%. «та» (вы), «не», «нас» (возраст) — частые калмыцкие слова, их в
# списке нет. На одном-двух словах доли ничего не значат: такой текст
# русским не считаем никогда.
RU_FUNC = frozenset(
    "и в на что с по как это для от к о из у за до но же бы ли то так его "
    "она он они мы вы ты я был была было были есть уже ещё еще или если чтобы "
    "когда только при над под без через все всё этот эта эти тот те мне "
    "меня вас их ему ей им который которая которые".split()
)

_WORD = re.compile(f"[{_CYR}]+(?:-[{_CYR}]+)*")


def is_russian(text: str) -> bool:
    toks = [t.lower() for t in _WORD.findall(normalize_lookalikes(text))]
    if len(toks) < 3:
        return False
    func = sum(t in RU_FUNC for t in toks) / len(toks)
    long = [t for t in toks if len(t) >= 3]
    vend = sum(t[-1] in "аеёиоуыэюяй" for t in long) / max(1, len(long))
    return func >= 0.12 or vend >= 0.55


# --------------------------------------------------------------------------
# Словарь и восстановление
# --------------------------------------------------------------------------
_C = "".join(sorted(CONS))
# Чего не бывает в правильном калмыцком слове: «ё», «яя», «юю», «йа»/«йу»
# в начале (пишут «я», «ю»), «э» не в начале слова (после согласной пишут
# «е»: в проверенном словаре модели таких слов ноль; «ээ» в начале —
# норма: ээҗ, ээм), согласная + я/ю в слове, которого нет в словаре (в
# словарных — норма: буудя, соляд).
_HARD_OOV = re.compile(f"ё|яя|юю|^й[ау]|(?<=[^-э])э|[{_C}][яю]")


@dataclass
class Detection:
    """Что сработало. flag — почему текст считается набранным без спецбукв
    ('' — не считается)."""

    flag: str = ""
    russian: bool = False
    reasons: Counter = field(default_factory=Counter)


@dataclass
class FixResult:
    text: str
    detection: Detection
    # (как было, как стало) — по одному на исправленное слово
    fixes: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.fixes)


class LetterFixer:
    def __init__(self, vocab: Dict[str, int], protected: Iterable[str] = (),
                 trusted: Iterable[str] = ()):
        # vocab: правильное слово -> частота; protected — русские слова,
        # которые встречаются в калмыцких текстах (гражданск, доктор): их
        # «исправлять» нельзя, хотя они похожи на калмыцкие без спецбукв;
        # trusted — слова проверенного словаря модели: годятся в основы,
        # даже если в корпусе встретились всего раз
        self.freq = vocab
        self.protected = set(protected)
        self.trusted = set(trusted)
        self.index: Dict[str, List[str]] = {}
        for v in vocab:
            self.index.setdefault(skeleton(v), []).append(v)
        self._build_suffixes()

    def _build_suffixes(self) -> None:
        """Окончания из самого словаря: слово = словарная основа + хвост.
        Считаются отдельно для основ каждого ряда: у твёрдой основы «-на»,
        у мягкой «-нә»."""
        counts: Counter = Counter()
        for v, n in self.freq.items():
            if "-" in v:
                continue
            for k in range(max(STEM_MIN, len(v) - SUFFIX_MAX), len(v)):
                stem = v[:k]
                if self._good_stem(stem):
                    counts[(vowel_row(stem), v[k:])] += n
        self.suffixes = {k: n for k, n in counts.items() if n >= SUFFIX_MIN_COUNT}
        self.suffix_index: Dict[str, Set[str]] = {}
        for _, x in self.suffixes:
            self.suffix_index.setdefault(skeleton(x), set()).add(x)

    # ------------------------------------------------------------ кандидаты
    def candidates(self, w: str, habits: Optional[Counter] = None):
        """[(слово, оценка, замены)] по убыванию оценки. w — строчными."""
        keys = _query_keys(w)
        if w.startswith(("йа", "йу")):
            # «Йамаран» — так пишут я/ю в начале слова
            alt = ("я" if w[1] == "а" else "ю") + w[2:]
            keys |= _query_keys(alt)
        else:
            alt = None
        seen = {}
        for k in keys:
            for v in self.index.get(k, ()):
                if v in seen:
                    continue
                best = None
                for src, extra in ((w, 0.0), (alt, 0.5)):
                    if src is None:
                        continue
                    r = channel(v, src)
                    if r and (best is None or r[0] + extra < best[0]):
                        best = (r[0] + extra, r[1])
                if best:
                    seen[v] = best
        out = []
        for v, (cost, subs) in seen.items():
            if habits:
                cost -= sum(
                    SUB_COST[a][b] * (1 - HABIT_DISCOUNT)
                    for a, b in subs if a in SUB_COST and habits[(a, b)]
                )
            score = math.log(self.freq[v] + 1) - COST_WEIGHT * cost
            out.append((v, score, subs))
        out.sort(key=lambda x: -x[1])
        return out

    # ------------------------------------------------------------ текст
    def detect(self, text: str) -> Detection:
        """Написан ли текст без спецбукв. Ничего не меняет."""
        d = Detection()
        raw = unicodedata.normalize("NFC", text)
        d.russian = is_russian(raw)
        if d.russian:
            return d
        d.reasons["двойник"] = sum(
            1 for t in _ANY_TOKEN.findall(raw) if _HAS_CYR.search(t) and _FOREIGN.search(t)
        )
        norm = normalize_lookalikes(raw).lower()
        toks = _WORD.findall(norm)
        n_special = sum(ch in SPECIAL for ch in norm)
        for w in toks:
            if w in self.freq or self.is_russian_word(w):
                continue
            if _HARD_OOV.search(w):
                d.reasons["жёсткий"] += 1
            c = self.candidates(w)
            if c and c[0][0] != w and SPECIAL & set(c[0][0]):
                d.reasons["словарь"] += 1
        n = len(toks)
        if d.reasons["двойник"] or d.reasons["жёсткий"]:
            d.flag = "hard"
        elif n_special == 0 and d.reasons["словарь"]:
            d.flag = "vocab"
        elif n_special == 0 and n >= 7:
            d.flag = "length"
        elif d.reasons["словарь"] >= 2 and d.reasons["словарь"] >= 0.15 * n:
            d.flag = "partial"
        return d

    def fix(self, text: str, force: bool = False) -> FixResult:
        """Исправить текст, если он написан без спецбукв. Правильный текст
        (спецбуквы на месте, триггеры молчат) и русский не трогаем.

        force=True — человек сам попросил проверить текст (режим /fix):
        незнакомые слова исправляются, даже если текст в целом написан
        правильно. Русский текст и словарные слова не трогаются и тут."""
        det = self.detect(text)
        norm = normalize_lookalikes(text)
        res = FixResult(text=norm, detection=det)
        if det.russian or not (det.flag or force):
            if norm != unicodedata.normalize("NFC", text):
                res.fixes = _diff_words(text, norm)
            return res

        tokens = list(_WORD.finditer(norm))
        # 1-й проход: однозначные слова показывают привычки автора
        habits: Counter = Counter()
        for m in tokens:
            w = m.group(0).lower()
            if w in self.freq or self.is_russian_word(w):
                continue
            c = self.candidates(w)
            if len(c) == 1 or (len(c) > 1 and c[0][1] - c[1][1] > 2):
                habits.update(s for s in c[0][2] if s[0] in SUB_COST)

        # 2-й проход: исправляем
        out, pos = [], 0
        for m in tokens:
            orig = m.group(0)
            w = orig.lower()
            new = w
            # В правильно набранном тексте (проверка по просьбе, force)
            # слово с заглавной не в начале предложения — скорее имя:
            # Кензеев, Юра, Сарангов. В тексте без спецбукв имена испорчены
            # так же, как всё остальное (Пюрвя), и их чиним.
            name = not det.flag and orig[:1].isupper() and not _sentence_start(norm, m.start())
            if w not in self.freq and not self.is_russian_word(w) and not name:
                new = self._restore(w, habits)
            out.append(norm[pos:m.start()])
            out.append(_match_case(orig, new) if new != w else orig)
            pos = m.end()
        out.append(norm[pos:])
        res.text = "".join(out)
        res.fixes = _diff_words(text, res.text)
        return res

    def _restore(self, w: str, habits: Counter) -> str:
        c = self.candidates(w, habits)
        if c:
            return c[0][0]
        if "-" in w:
            # составное слово: чиним части по отдельности
            parts = [p if p in self.freq else self._restore(p, habits) for p in w.split("-")]
            return "-".join(parts)
        return self._restore_split(w, habits)

    def is_russian_word(self, w: str) -> bool:
        """Русское слово из калмыцких текстов — само или с калмыцким
        окончанием до 4 букв: машин-ас, поезд-ар, Сталинград-т."""
        if w in self.protected:
            return True
        return any(w[:k] in self.protected for k in range(max(4, len(w) - 4), len(w)))

    def _good_stem(self, stem: str) -> bool:
        return self.freq.get(stem, 0) >= STEM_MIN_FREQ or stem in self.trusted

    def _restore_split(self, w: str, habits: Counter) -> str:
        """Основа из словаря + окончание, согласованное с её рядом."""
        best, best_score = w, None
        for k in range(max(STEM_MIN, len(w) - SUFFIX_MAX), len(w)):
            s, x = w[:k], w[k:]
            stems = [c for c in self.candidates(s, habits) if self._good_stem(c[0])][:3]
            if not stems:
                continue
            tails = {t for key in _query_keys(x) for t in self.suffix_index.get(key, ())}
            for stem, stem_score, _ in stems:
                row = vowel_row(stem)
                for t in tails:
                    n = self.suffixes.get((row, t))
                    r = channel(t, x) if n else None
                    if r is None:
                        continue
                    score = stem_score + math.log(n) - COST_WEIGHT * r[0] - SPLIT_PENALTY
                    if best_score is None or score > best_score:
                        best, best_score = stem + t, score
        return best


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \t«\"'(—–-")
    return not before or before[-1] in ".!?\n…"


def _match_case(orig: str, new: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return new.upper()
    if orig[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _diff_words(before: str, after: str) -> List[Tuple[str, str]]:
    a = _ANY_TOKEN.findall(unicodedata.normalize("NFC", before))
    b = _ANY_TOKEN.findall(after)
    if len(a) != len(b):
        return [(before, after)] if before != after else []
    return [(x, y) for x, y in zip(a, b) if x != y]


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------
def load_vocab(path) -> Tuple[Dict[str, int], Set[str], Set[str]]:
    """Файл: слово<TAB>частота<TAB>d|k|r. d — слово из проверенного словаря
    модели, k — из корпуса, r — русское слово из калмыцких текстов, его не
    трогаем. -> (частоты, русские, проверенные)."""
    vocab, protected, trusted = {}, set(), set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            word, n, kind = line.rstrip("\n").split("\t")
            if kind == "r":
                protected.add(word)
            else:
                vocab[word] = int(n)
                if kind == "d":
                    trusted.add(word)
    return vocab, protected, trusted


_fixer: Optional[LetterFixer] = None


def get_fixer() -> LetterFixer:
    """Общий экземпляр (словарь грузится при первом обращении)."""
    global _fixer
    if _fixer is None:
        path = os.environ.get("LETTERS_VOCAB_PATH") or DEFAULT_VOCAB
        _fixer = LetterFixer(*load_vocab(path))
    return _fixer


def fix_text(text: str, force: bool = False) -> FixResult:
    return get_fixer().fix(text, force=force)


def detect(text: str) -> Detection:
    return get_fixer().detect(text)
