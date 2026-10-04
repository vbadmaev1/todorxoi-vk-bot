# -*- coding: utf-8 -*-
"""
broken_words.py — слово, которое распознавание разорвало на два.

В рукописном и старопечатном тексте перо отрывали посреди слова: между
ǰi и kü в örgöǰikü остаётся крошечный просвет. Модель распознавания
училась на шрифтах, где такой просвет — узкий пробел перед суффиксом, и
ставит пробел: «örgöǰi kü», «ged eq», «ye ke», «d ü».

По картинке это не решить. У отрыва пера нет ни одной пустой полоски
поперёк столбца (штрихи перекрываются), но в шрифтах так же выглядят и
настоящие узкие пробелы (972 из 2808 на синтетике), а в шрифте zakaa —
даже 4 % обычных пробелов: хвосты букв заходят под соседнее слово.

Решает язык. Куски a и b склеиваются, если слитное ab встречается в
корпусе не реже, чем более редкий из кусков:

    ged (5)    + eq (0)     -> gedeq (2881)     склеить
    d (0)      + ü (4)      -> dü (9057)        склеить
    örgöǰi (173) + kü (2)   -> örgöǰikü (2)     склеить
    erdeni (183) + dü (9057) -> erdenidü (0)    оставить

На отложенных предложениях корпуса правило склеило бы 29 швов из 30 811
(0.09 %), и почти все такие пары — куски одного кириллического слова
(amita ni, nöl ügei): на кириллицу это не влияет. Незнакомое слитное
слово (старая орфография) ничего не склеивает — остаётся как было.

Второе правило — для слов, которых в корпусе нет (старая орфография, ошибка
в букве): d uu -> duu, köüked iain -> köükediain, be yeni -> beyeni. Нужны
все три условия:
  * узкий пробел, у которого на картинке нет ни одной пустой полоски
    поперёк столбца (штрихи перекрываются — отрыв пера);
  * правый кусок редкий (встречается в корпусе реже RARE раз: yeni, ü,
    iain) или левый — незнакомый обрывок в 1–2 буквы (d uu). Незнакомая
    основа перед частым суффиксом (dobuyin ni, nasun ni) — это обычный
    узкий пробел редкого слова, его не трогаем;
  * посимвольная языковая модель по словам корпуса (5-граммы, Witten-Bell)
    считает ab как одно слово правдоподобнее, чем a и b по отдельности, с
    запасом PEN_LIFT_GAIN: d+uu +12.6, köüked+iain +12.7, mend+ü +10.9,
    be+yeni +10.1; настоящий суффикс kele+bēr +3.7, частица ene+le +9.1.
Замер на синтетике (2488 столбцов шести шрифтов): без условий на куски и
языковой модели правило портило 157 столбцов, с ними — ни одного сверх
словарного правила (6 столбцов, почти все — суффиксы одного кириллического
слова). Языковая модель строится при прогреве: ~1.3 с, ~30 МБ.

Частоты — model/todo_words.tsv.gz, собирает tools/build_todo_words.py.
"""

import gzip
import math
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .translit_todo import todo_to_translit

_HERE = Path(__file__).resolve().parent
_DEFAULT_WORDS = _HERE.parent / "model" / "todo_words.tsv.gz"

_GAP = re.compile("([  ]+)")

_FREQ: Optional[Dict[str, int]] = None
_LOCK = threading.Lock()


def words_path() -> str:
    return os.environ.get("TODO_WORDS_PATH") or str(_DEFAULT_WORDS)


def get_freq() -> Dict[str, int]:
    global _FREQ
    if _FREQ is None:
        with _LOCK:
            if _FREQ is None:
                freq = {}
                with gzip.open(words_path(), "rt", encoding="utf-8") as f:
                    for line in f:
                        word, n = line.rstrip("\n").split("\t")
                        freq[word] = int(n)
                _FREQ = freq
    return _FREQ


# буквы у шва: хвост левого куска и начало правого (знаки препинания вокруг
# не мешают: «örgöǰi kü᠂» — тоже разрыв)
_TAIL = re.compile("[\u1820-\u18aa]+$")
_HEAD = re.compile("^[\u1820-\u18aa]+")


NNBSP = "\u202f"
# \u0417\u043d\u0430\u043a \u0434\u043e\u043b\u0433\u043e\u0442\u044b. \u0412 \u0447\u0430\u0441\u0442\u0438 \u0448\u0440\u0438\u0444\u0442\u043e\u0432 (\u0441\u0438\u043d\u044c\u0446\u0437\u044f\u043d\u0441\u043a\u0438\u0435 \u043a\u043d\u0438\u0433\u0438) \u043e\u043d \u043d\u0430\u0440\u0438\u0441\u043e\u0432\u0430\u043d \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u044b\u043c \u0448\u0442\u0440\u0438\u0445\u043e\u043c \u0441\u0431\u043e\u043a\u0443, \u0438 \u043c\u043e\u0434\u0435\u043b\u044c
# \u0441\u0442\u0430\u0432\u0438\u0442 \u043f\u0435\u0440\u0435\u0434 \u043d\u0438\u043c \u043f\u0440\u043e\u0431\u0435\u043b: baqta :qsan, t\u0101larxa :d. \u0412 \u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0435 \u0440\u0443\u043a\u043e\u043f\u0438\u0441\u0435\u0439 \u0438 \u0432 \u0441\u0438\u043d\u0442\u0435\u0442\u0438\u043a\u0435 \u043f\u0440\u043e\u0431\u0435\u043b\u0430 \u043f\u0435\u0440\u0435\u0434
# \u043d\u0438\u043c \u043d\u0435\u0442 \u043d\u0438 \u0440\u0430\u0437\u0443 \u2014 \u0442\u0430\u043a\u043e\u0439 \u0448\u043e\u0432 \u0441\u043a\u043b\u0435\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u0432\u0441\u0435\u0433\u0434\u0430.
LONG_VOWEL = "\u1843"
# на сколько (натуральный логарифм) слитное слово должно быть правдоподобнее
# двух отдельных, чтобы склеить шов-отрыв пера без слитного слова в корпусе
PEN_LIFT_GAIN = 10.0
# То же для узкого пробела с просветом на картинке. На мелком скане (синьцзянская книга, столбец ~28 px)
# бледный соединительный штрих после масштабирования светлее порога INK, и просвет «есть» и посреди слова:
# kü ndüdkel, dömü rgeǰi, bayix u. Запас нужен больше: на 10 разворотах книги склеилось ~50 швов, почти все
# верно; на синтетике (3422 страницы) — 6 лишних (temne+kü), на настоящих строках — ни одного.
PEN_LIFT_GAIN_LOOSE = 14.0
# кусок, встреченный в корпусе реже этого, — скорее обрывок слова (ü, dür)
RARE = 5
LM_ORDER = 5


class CharLM:
    """Посимвольная n-граммная модель слов (Witten-Bell). Учится на типах
    слов, а не на употреблениях: иначе частые суффиксы (ni, bēr) задавили
    бы всё остальное."""

    def __init__(self, words, order: int = LM_ORDER):
        self.order = order
        cnt, ctx_n = {}, {}
        for w in words:
            s = "^" * (order - 1) + w + "$"
            for i in range(order - 1, len(s)):
                for k in range(1, order + 1):
                    c = s[i - k + 1:i]
                    key = c + "\t" + s[i]
                    cnt[key] = cnt.get(key, 0) + 1
                    ctx_n[c] = ctx_n.get(c, 0) + 1
        ctx_t = {}
        for key in cnt:
            c = key.split("\t", 1)[0]
            ctx_t[c] = ctx_t.get(c, 0) + 1
        self.cnt, self.ctx_n, self.ctx_t = cnt, ctx_n, ctx_t
        self.vocab = len({ch for w in words for ch in w}) + 1

    def _prob(self, s: str, i: int, k: int) -> float:
        if k == 0:
            return 1.0 / self.vocab
        c = s[i - k + 1:i]
        lower = self._prob(s, i, k - 1)
        n = self.ctx_n.get(c, 0)
        if not n:
            return lower
        lam = n / (n + self.ctx_t[c])
        return lam * self.cnt.get(c + "\t" + s[i], 0) / n + (1 - lam) * lower

    def logp(self, word: str) -> float:
        s = "^" * (self.order - 1) + word + "$"
        return sum(math.log(self._prob(s, i, self.order)) for i in range(self.order - 1, len(s)))

    def gain(self, a: str, b: str) -> float:
        """Насколько ab как одно слово правдоподобнее, чем a и b по отдельности."""
        return self.logp(a + b) - self.logp(a) - self.logp(b)


_LM: Optional[CharLM] = None


def get_lm() -> CharLM:
    global _LM
    if _LM is None:
        freq = get_freq()
        with _LOCK:
            if _LM is None:
                _LM = CharLM(freq)
    return _LM


def is_pen_lift(a: str, b: str, freq: Dict[str, int], gain: float = PEN_LIFT_GAIN) -> bool:
    """Шов без просвета на картинке: склеить ли a и b, когда слитного слова
    в корпусе нет (условия — в начале файла)."""
    fa, fb = freq.get(a, 0), freq.get(b, 0)
    fragment = fb < RARE or (fa == 0 and len(a) <= 2)
    return fragment and get_lm().gain(a, b) > gain


def should_join(a: str, b: str, ab: str, freq: Dict[str, int]) -> bool:
    """a, b — транслитерация кусков у шва, ab — слитного слова (считается
    отдельно: x/k и g/γ зависят от соседней буквы)."""
    n = freq.get(ab, 0)
    return n > 0 and n >= min(freq.get(a, 0), freq.get(b, 0))


def _goes_right(head: str, right_word: str, ab: str, freq: Dict[str, int]) -> bool:
    """Обрывок посередине («mori d ēn», «bi d ü»): склеить его скорее с
    правым соседом? Да, если весь кусок — буквы, справа он складывается в
    слово корпуса и это слово частотнее, чем слитное с левым: dēn (2012)
    важнее morid (198), dü (9057) — bid (5)."""
    nxt = _HEAD.match(right_word)
    if not nxt or not _HEAD.fullmatch(head):
        return False
    b, c = todo_to_translit(head), todo_to_translit(nxt.group())
    bc = todo_to_translit(head + nxt.group())
    return should_join(b, c, bc, freq) and freq.get(bc, 0) > freq.get(ab, 0)


def join_mask(todo: str, tight: Optional[Sequence[bool]] = None) -> List[bool]:
    """Какие символы строки оставить (False — промежуток внутри слова).

    tight[k] — у узкого пробела на месте k на картинке нет ни одной пустой
    полоски поперёк столбца (см. core/ocr.py, Model.read). Такой шов
    склеивается и без слитного слова в корпусе — по is_pen_lift; узкий пробел
    с просветом — тоже, но с запасом PEN_LIFT_GAIN_LOOSE."""
    keep = [True] * len(todo)
    gaps = list(_GAP.finditer(todo))
    if not gaps:
        return keep
    freq = get_freq()
    words = [todo[:gaps[0].start()]] + [
        todo[g.end():(gaps[k + 1].start() if k + 1 < len(gaps) else len(todo))]
        for k, g in enumerate(gaps)
    ]
    left = words[0]
    for k, gap in enumerate(gaps):
        word = words[k + 1]
        tail, head = _TAIL.search(left), _HEAD.match(word)
        join = False
        if tail and head and head.group()[0] == LONG_VOWEL:
            join = True                                           # со знака долготы слово не начинается
        elif tail and head:
            a, b = todo_to_translit(tail.group()), todo_to_translit(head.group())
            ab = todo_to_translit(tail.group() + head.group())
            right = (freq.get(b, 0) == 0 and k + 1 < len(gaps)
                     and _goes_right(head.group(), words[k + 2], ab, freq))
            if not right:
                narrow = all(todo[i] == NNBSP for i in range(gap.start(), gap.end()))
                no_blank = tight is not None and all(tight[i] for i in range(gap.start(), gap.end()))
                join = should_join(a, b, ab, freq) or (
                    narrow and tight is not None
                    and is_pen_lift(a, b, freq, PEN_LIFT_GAIN if no_blank else PEN_LIFT_GAIN_LOOSE)
                )
        if join:
            for i in range(gap.start(), gap.end()):
                keep[i] = False
            left += word
        else:
            left = word
    return keep


def join_broken_words(todo: str, tight: Optional[Sequence[bool]] = None) -> str:
    """Строка тодо бичиг (столбец) -> та же строка без разрывов внутри слов.

    Решение принимается по транслитерации, но убирается сам промежуток в
    тодо бичиг: и транслитерация, и кириллица дальше строятся уже из
    исправленной строки."""
    return "".join(ch for ch, k in zip(todo, join_mask(todo, tight)) if k)


def warmup() -> int:
    get_lm()
    return len(get_freq())
