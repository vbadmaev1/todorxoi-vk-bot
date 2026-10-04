# -*- coding: utf-8 -*-
"""
ocr.py — распознавание тодо бичиг с фото: картинка -> столбцы -> текст.

  1. page_layout.split_page режет страницу на столбцы (выпрямляет наклон,
     выкидывает колонтитулы и пыль) — подробности в самом модуле.
  2. CRNN + CTC (model/todo_ocr_int8.onnx) читает каждый столбец: столбец
     поворачивается, приводится к высоте 64 и идёт в сеть. Модель int8 под
     onnxruntime: 4 МБ, torch не нужен; на CPU ~1 с на полную страницу в
     один поток. Как она получена из обученной — tools/export_ocr_onnx.py.
  2а. Слова, разорванные отрывом пера (örgöǰi kü -> örgöǰikü), склеиваются
     по частотам слов корпуса и по тому, есть ли на картинке просвет в
     узком пробеле — core/broken_words.py, вызывается из Model.read.
  3. Транслитерация — тем же todo_to_translit, что и в текстовых режимах;
     перед ней убираются висячие узкие пробелы, знаки — в латинские.
  3а. Кириллица — core/to_cyrillic.py: сегментатор решает, какие слова
     тодо бичиг складываются в одно кириллическое (oron du -> орнд), модель
     переводит. ~15 мс на столбец; без её файлов ответ просто без кириллицы.
  4. Картинка-проверка: присланная картинка (выпрямленная, в своих цветах)
     с рамками и номерами столбцов. Видно, как бот разрезал страницу, и
     какая строка ответа какому столбцу соответствует. Стоит ~50 мс и один
     JPEG.

Всё синхронное и счётное — в боте звать через asyncio.to_thread и по одной
картинке за раз (см. bot/handlers/ocr.py): модель с библиотеками держит
~95 МБ, полная страница — ещё ~160 МБ на время чтения, две параллельно —
вдвое больше.
"""

import io
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .punctuation import DEFAULT_PUNCT, apply_punctuation
from .translit_todo import normalize_ocr, todo_to_translit

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = _HERE.parent / "model" / "todo_ocr_int8.onnx"

# Больше этого по длинной стороне картинка уменьшается: память разметки
# растёт с площадью, а точность на сканах одинаковая от 1000 до 2700 px.
MAX_SIDE = 2500
# Столбцов в сеть за один прогон: 2 — пик ~280 МБ на страницу, 8 — ~670 МБ
# при той же скорости и точности.
MAX_BATCH = 2
# Картинка-проверка крупнее не нужна: Telegram всё равно пережмёт.
OVERLAY_SIDE = 1600
# Средняя уверенность модели (вероятность выбранного символа по всем
# непустым шагам CTC) ниже этого — вероятно, на фото не тодо бичиг или
# текст плохо читается. Замерено: страницы тодо бичиг у Позднеева 0.95–0.97
# (и так же после сжатия до 1280 px в JPEG), синтетика 0.96–1.00; русские
# страницы той же книги — почти всегда ни одной буквы, а где что-то нашлось,
# 0.43–0.93 (0.93 — одна буква на всю страницу, её ловит MIN_LETTERS).
LOW_CONFIDENCE = 0.9
# Слово на краю столбца с надписи на предмете с медианной уверенностью ниже
# этого — не текст, а складка, фон или отверстие (см. trim_unsure). Настоящие
# слова на фото кулона и татуировки: 1.0; мусор: 0.52–0.88.
UNSURE_WORD = 0.9
# Столбец — мусор, если средняя уверенность букв ниже JUNK_MEAN и неуверенных (< UNSURE_WORD) не меньше
# JUNK_UNSURE_SHARE (см. _junk_column). На 10 разворотах книги-скана так уходят только края, корешок и
# размытые столбцы; на настоящих строках рукописей CER test не меняется.
JUNK_MEAN = 0.8
JUNK_UNSURE_SHARE = 0.5
# Отброшенный столбец, в котором модель прочитала столько букв (и он размером как соседние), — размытый
# текст: о нём говорим в ответе. Пустое место с просвечивающей обратной стороной даёт до 6 «букв».
UNREADABLE_MIN_LETTERS = 7
# Повёрнутое прочтение берём, если в нём уверенно прочитано в ROTATION_GAIN раз больше, чем в прямом, и
# средняя уверенность не ниже ROTATED_MIN_CONFIDENCE. Пустотная рукопись: перевёрнутый лист 105 -> 427
# (уверенность 0.83 -> 0.96), титульная строка поперёк кадра 0 -> 36 (0.89); мусор в повороте — до 0.83.
ROTATION_GAIN = 1.5
ROTATED_MIN_CONFIDENCE = 0.85
# Страницу перечитываем перевёрнутой, если её длинные столбцы вверх ногами читаются увереннее на столько
# (см. _flip_gain): у прямых страниц разница от -0.04 и ниже, у перевёрнутых +0.09…+0.16.
FLIP_GAIN = 0.05
# Прочитано меньше стольких букв — пробуем и повороты на бок: страница на боку даёт горстку «букв» с обычной
# уверенностью (3 буквы, 0.94), а не пустоту.
FEW_LETTERS = 10
# С вырезки на фото предмета меньше букв не принимаем: после обрезки
# неуверенного от складок и узоров остаются уверенные огрызки в 1–2 буквы
# («xa» с кирпичной стены), а настоящая надпись — хотя бы слово.
PLATE_MIN_LETTERS = 3
# Меньше букв на всю картинку — считаем, что тодо бичиг на ней нет.
MIN_LETTERS = 2

_BOX_COLORS = ((230, 40, 40), (30, 110, 235))   # соседние столбцы — разным цветом
_UNREAD_COLOR = (130, 130, 130)                  # размытый столбец, которого нет в тексте

_GAP = " ᠂᠃︱︖︕"                  # пробел и знаки препинания из алфавита модели
_NOT_LETTERS = set(_GAP + "\u202f")
_LONG_VOWEL = "\u1843"             # знак долготы: сам по себе не буква (в транслитерации «:»)
# Узкий неразрывный пробел (U+202F) отделяет суффикс и стоит между буквами.
# Рядом с пробелом или знаком он смысла не имеет, а модель ставит его на
# широких промежутках между словами (в книгах столбцы выключены по высоте,
# и промежутки растянуты). В транслитерации он стал бы висячим дефисом:
# «talaa- ǰiliyaiǰi-».
_LOOSE_NNBSP = re.compile(f"\u202f+(?=[{_GAP}]|$)|(?<=[{_GAP}])\u202f+|^\u202f+")
# На странице знаки препинания — настоящие монгольские (᠂ ᠃), а
# todo_to_translit знает только вертикальные формы из текстового режима
# (︐ ︒). В транслитерации — обычные латинские знаки. У бирги латинской
# пары нет, четыре точки — конец текста, то есть точка.
_PUNCT_TO_LATIN = str.maketrans(
    {"᠂": ",", "᠃": ".", "︖": "?", "︕": "!", "︱": "—", "᠀": None, "᠅": "."}
)


# Слово тодо бичиг без гласной — почти всегда не слово, а то, чего нет в
# алфавите модели: скобки 『』（）, цифры, латиница, «+», китайские знаки
# препинания; модель читает их как одну-две согласные (q, l, p, m, dm, yt).
# В разметке рукописей таких «слов» 4 на 80 тыс., в корпусе 0.05%. По
# уверенности модели их не отличить: мусорную q она видит с 1.0, а
# настоящее ü — бывает и с 0.6. Обрывки разорванного слова (d uu) к этому
# времени уже склеены (Model.read), знаки препинания остаются.
_TODO_VOWELS = set("ᠠᡄᡅᡆᡇᡈᡉ")
_WORD = re.compile("[^ \u202f]+")
_LETTER = re.compile("[\u1820-\u18aa]")
_MARKS = re.compile("[\u180b-\u180f\u1820-\u18aa]")              # буквы и селекторы вариантов


def _drop_vowelless(match) -> str:
    word = match.group()
    letters = _LETTER.findall(word)
    if not letters or len(letters) > 2 or any(ch in _TODO_VOWELS for ch in letters):
        return word
    return _MARKS.sub("", word)


def tidy(text: str) -> str:
    text = _WORD.sub(_drop_vowelless, text)
    text = _LOOSE_NNBSP.sub("", text)
    text = re.sub("\u202f{2,}", "\u202f", text)
    return re.sub(" {2,}", " ", text).strip(" ")


# Узкий пробел в транслитерации — дефис (kele-bēr). Модель распознавания
# отличить его от обычного пробела на фото не может: она училась на разметке,
# где дефисом из словаря отмечены куски одного кириллического слова, и ставит
# узкий пробел почти в каждый промежуток — на странице с обычными пробелами
# выходило «dēre-ügei-üzüülüqči-burxan-erdeni». Поэтому в транслитерации
# распознанного текста любой промежуток — пробел. Какие куски на самом деле
# одно кириллическое слово, решает уже не OCR. В тодо бичиг узкий пробел
# остаётся: там он виден только как чуть меньший промежуток.
def to_translit(todo: str) -> str:
    return todo_to_translit(todo.replace(" ", " ")).translate(_PUNCT_TO_LATIN)


def to_cyrillic_columns(translit: list) -> list:
    """Столбцы транслитерации -> столбцы кириллицы, одна строка к одной.
    Кириллица — дополнение к ответу: если её модели нет или она упала,
    распознавание всё равно отдаёт тодо бичиг и транслитерацию."""
    try:
        from .to_cyrillic import to_cyrillic
        cyrillic = to_cyrillic("\n".join(translit)).split("\n")
    except Exception:
        log.exception("кириллица для распознанного текста не получилась")
        return []
    return cyrillic if len(cyrillic) == len(translit) else []


class OcrError(Exception):
    """Ошибка, текст которой можно показать пользователю как есть."""


class OcrUnavailable(OcrError):
    """Нет onnxruntime или файла модели: бот работает, но фото не читает."""


@dataclass
class OcrResult:
    columns: list                          # тодо бичиг, по строке на столбец
    translit_columns: list
    cyrillic_columns: list = field(default_factory=list)   # пусто — не получилось
    confidence: float = 1.0
    overlay: Optional[bytes] = None        # JPEG с рамками столбцов
    overlay_size: tuple = (0, 0)
    angle: float = 0.0                     # на сколько градусов выпрямили
    elapsed_ms: float = 0.0
    steps_ms: dict = field(default_factory=dict)
    unreadable: int = 0                    # столбцов с текстом, который не разобрать (размыт)

    @property
    def todo(self) -> str:
        return "\n".join(self.columns)

    @property
    def translit(self) -> str:
        return "\n".join(self.translit_columns)

    @property
    def cyrillic(self) -> str:
        return "\n".join(self.cyrillic_columns)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


# ------------------------------------------------------------------ модель

def column_line(column: Image.Image, img_h: int = 64) -> np.ndarray:
    """Столбец -> горизонтальная строка высотой img_h, чернила = 1.
    То же преобразование, что при обучении: поворот, масштаб, ширина
    кратна 4 (сеть сжимает ширину вчетверо)."""
    im = column.convert("L").rotate(90, expand=True)
    w, h = im.size
    im = im.resize((max(8, round(w * img_h / h)), img_h), Image.BILINEAR)
    a = (255 - np.asarray(im, dtype=np.float32)) / 255.0
    pad = (-a.shape[1]) % 4
    return np.pad(a, ((0, 0), (0, pad))) if pad else a


class Model:
    def __init__(self, path: str, threads: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OcrUnavailable("не установлен onnxruntime") from exc
        if not Path(path).exists():
            raise OcrUnavailable(f"нет файла модели: {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.chars = json.loads(meta["chars"])     # индекс k в выходе = chars[k - 1]; 0 = blank
        self.img_h = int(meta["img_h"])
        self.path = path

    def read(self, columns: list, char_probs: bool = False):
        """-> (строки, средняя уверенность по всем непустым шагам)
        или, с char_probs, (строки, уверенность, [уверенности символов строки ...])."""
        lines = [column_line(c, self.img_h) for c in columns]
        order = np.argsort([a.shape[1] for a in lines])   # близкие по ширине — в один батч
        texts = [""] * len(lines)
        probs = [[] for _ in lines]
        conf_sum, conf_n = 0.0, 0
        for k in range(0, len(order), MAX_BATCH):
            idx = order[k:k + MAX_BATCH]
            width = max(lines[i].shape[1] for i in idx)
            x = np.zeros((len(idx), 1, self.img_h, width), np.float32)
            for j, i in enumerate(idx):
                x[j, 0, :, :lines[i].shape[1]] = lines[i]
            logits = self.sess.run(None, {"image": x})[0]
            for j, i in enumerate(idx):
                steps = logits[j, :lines[i].shape[1] // 4]
                best = steps.argmax(-1)
                # CTC: схлопываем повторы, убираем blank
                keep = (best != 0) & np.concatenate([[True], best[1:] != best[:-1]])
                texts[i] = "".join(self.chars[c - 1] for c in best[keep])
                p = np.exp(steps - steps.max(-1, keepdims=True))
                p = (p / p.sum(-1, keepdims=True)).max(-1)
                probs[i] = p[keep].tolist()
                texts[i], probs[i] = _join_broken(
                    texts[i], probs[i], tight_gaps(texts[i], np.nonzero(keep)[0], lines[i]))
                conf_sum += float(p[best != 0].sum()); conf_n += int((best != 0).sum())
        conf = conf_sum / conf_n if conf_n else 0.0
        return (texts, conf, probs) if char_probs else (texts, conf)


# Чернила в строке для сети — от 0 до 1; ниже этого — фон (сглаживание краёв).
INK = 0.35


def tight_gaps(text: str, frames, line: np.ndarray) -> list:
    """Для каждого узкого пробела: True, если между соседними буквами нет ни
    одной пустой полоски поперёк столбца. Так выглядит отрыв пера посреди
    слова: штрихи до и после перекрываются. frames — шаг CTC каждого символа
    (шаг = 4 пикселя строки)."""
    blank = (line > INK).sum(0) == 0
    tight = [False] * len(text)
    for i, ch in enumerate(text):
        if ch == "\u202f" and 0 < i < len(text) - 1:
            tight[i] = not blank[frames[i - 1] * 4:frames[i + 1] * 4 + 4].any()
    return tight


def _join_broken(text: str, probs: list, tight: list):
    """Убрать промежутки внутри разорванных слов (core/broken_words.py) и их
    уверенности. Без файла частот строка остаётся как есть."""
    try:
        from .broken_words import join_mask
        mask = join_mask(text, tight)
    except Exception:
        log.exception("склейка разорванных слов не получилась")
        return text, probs
    return ("".join(c for c, k in zip(text, mask) if k),
            [q for q, k in zip(probs, mask) if k])


_model: Optional[Model] = None
_model_lock = threading.Lock()


def model_path() -> str:
    return os.environ.get("OCR_MODEL_PATH") or str(_DEFAULT_MODEL)


def get_model() -> Model:
    """Загруженная модель (грузится при первом обращении)."""
    global _model
    with _model_lock:
        if _model is None:
            threads = int(os.environ.get("OCR_THREADS") or 0) or min(4, os.cpu_count() or 1)
            _model = Model(model_path(), threads)
        return _model


def _split_page():
    try:
        from .page_layout import split_page
    except ImportError as exc:                   # разметке нужен scipy
        raise OcrUnavailable(f"не хватает зависимости: {exc.name}") from exc
    return split_page


def trim_unsure(text: str, probs: list):
    """Столбец с надписи на предмете -> (текст без неуверенных слов по краям, уверенности его букв);
    ("", []) — если неуверен весь.

    На странице всё в кадре — текст. На фото предмета в вырезку попадают складки кожи, кусок фона,
    отверстие под цепочку, и модель читает их как короткие «слова» по краям столбца или как
    отдельный столбец. Настоящие буквы она видит с уверенностью ~1.0, такие — 0.5–0.9. Судим по
    медиане слова: одна сомнительная буква посреди уверенного слова его не выкидывает."""
    words, cur = [], []
    for ch, p in zip(text, probs):
        if ch == " ":
            if cur:
                words.append(cur); cur = []
        else:
            cur.append((ch, p))
    if cur:
        words.append(cur)
    sure = [float(np.median([p for _, p in w])) >= UNSURE_WORD for w in words]
    while words and not sure[0]:
        words.pop(0); sure.pop(0)
    while words and not sure[-1]:
        words.pop(); sure.pop()
    while words and all(ch in _GAP for ch, _ in words[0]):
        words.pop(0)
    while words and words[-1] and words[-1][-1][0] in _GAP and words[-1][-1][1] < UNSURE_WORD:
        words[-1].pop()                                           # «︕» из складки, прилипший к последнему слову
        if not words[-1]:
            words.pop()
    return " ".join("".join(ch for ch, _ in w) for w in words), [p for w in words for _, p in w]


def _junk_column(text: str, probs: list) -> bool:
    """Столбец, который не текст: на фото здания или улицы край стены, рама окна, перила тоже дают
    «столбцы». Модель читает их как 0–2 знака: «—», «:::m», «no,» — и если букв две, то неуверенно.
    Настоящее короткое слово из двух букв она видит уверенно, его не трогаем."""
    p = [q for ch, q in zip(text, probs) if ch not in _NOT_LETTERS and ch != _LONG_VOWEL]
    if len(p) < 2 or (len(p) < 3 and float(np.median(p)) < UNSURE_WORD):
        return True
    # Столбец, где модель не уверена почти ни в чём: на скане разворота — обрез книги, тень корешка,
    # пустое место с просвечивающим текстом обратной стороны, номер страницы, а ещё столбцы у корешка
    # не в фокусе. Модель читает их как «rooml muǰγoǰozoolrz», «ooa zen eno yer»; повысить резкость
    # или контраст не помогает. Настоящие столбцы, даже бледные, — от 0.85 в среднем.
    return float(np.mean(p)) < JUNK_MEAN and float(np.mean(np.array(p) < UNSURE_WORD)) >= JUNK_UNSURE_SHARE


def _unreadable_boxes(todo: list, boxes: list, keep: list) -> list:
    """Рамки отброшенных столбцов, которые похожи на настоящие, но размытые: модель прочитала хотя бы
    UNREADABLE_MIN_LETTERS букв, а толщиной и длиной столбец как соседние. Край книги, тень корешка и
    номер страницы отличаются размером: во всю высоту, в несколько столбцов шириной или крошечные. На
    10 разворотах книги так отмечаются ровно размытые столбцы у корешка и размытый заголовок."""
    kept = [b for b, k in zip(boxes, keep) if k]
    if len(kept) < 3:
        return []
    w = float(np.median([b[2] - b[0] for b in kept]))
    h = max(b[3] - b[1] for b in kept)
    out = []
    for t, b, k in zip(todo, boxes, keep):
        letters = sum(ch not in _NOT_LETTERS for ch in t)
        if not k and letters >= UNREADABLE_MIN_LETTERS and 0.6 * w <= b[2] - b[0] <= 1.6 * w \
                and 2 * w <= b[3] - b[1] <= 1.15 * h:
            out.append(b)
    return out


def _flip_gain(model, cols: list) -> float:
    """Насколько модель увереннее читает самые длинные столбцы, перевернув их на 180° (> 0 — страница,
    похоже, вверх ногами). Перевёрнутая страница — те же столбцы, только каждый вверх ногами и в обратном
    порядке, так что хватает четырёх столбцов вместо второго прочтения всей страницы. Настоящие строки
    рукописей: всегда < -0.04; разворот книги — от -0.04 до -0.15; лист пустотной рукописи вверх ногами +0.16."""
    big = sorted(cols, key=lambda c: -c.height)[:4]
    if not big:
        return 0.0
    _, _, up = model.read(big, char_probs=True)
    _, _, down = model.read([c.rotate(180) for c in big], char_probs=True)
    up, down = [q for p in up for q in p], [q for p in down for q in p]
    return float(np.mean(down)) - float(np.mean(up)) if up and down else 0.0


def _margin_columns(boxes: list) -> list:
    """Для каждого столбца: True, если это колонтитул или номер страницы — короткий (ниже двух толщин
    столбца) и целиком ниже или выше всех нормальных столбцов. Цифры номера стоят врозь и дают по
    «столбцу» на цифру: «119» читалось как «bomi du». Короткий последний столбец стихотворения стоит
    в одном ряду с остальными — его не трогаем."""
    if len(boxes) < 3:
        return [False] * len(boxes)
    w = float(np.median([b[2] - b[0] for b in boxes]))
    tall = [b for b in boxes if b[3] - b[1] >= 2 * w]
    if not tall:
        return [False] * len(boxes)
    top, bottom = min(b[1] for b in tall), max(b[3] for b in tall)
    return [b[3] - b[1] < 2 * w and (b[1] >= bottom or b[3] <= top) for b in boxes]


def _read_page(page, split_page, model, trim=False):
    """Картинка -> (столбцы тодо, уверенность, число букв, отладка разметки) или None, если столбцов нет.
    trim — надпись на предмете: неуверенные слова по краям и неуверенные столбцы убираются.
    В отладке «unread» — рамки размытых столбцов, которые пришлось выкинуть (см. _unreadable_boxes)."""
    cols, dbg = split_page(page, debug=True)
    if not cols:
        return None
    todo, confidence, probs = model.read(cols, char_probs=True)
    keep = [not _junk_column(t, p) for t, p in zip(todo, probs)]
    dbg["unread"] = [] if trim else _unreadable_boxes(todo, dbg["boxes"], keep)
    keep = [k and not m for k, m in zip(keep, _margin_columns(dbg["boxes"]))]
    if not trim:
        dbg["flip_gain"] = _flip_gain(model, [c for c, k in zip(cols, keep) if k])
    if not all(keep):
        todo = [t for t, k in zip(todo, keep) if k]
        probs = [p for p, k in zip(probs, keep) if k]
        dbg["boxes"] = [b for b, k in zip(dbg["boxes"], keep) if k]
        left = [q for p in probs for q in p]                      # уверенность — по тому, что осталось
        confidence = float(np.mean(left)) if left else 0.0
    if trim:
        kept = [trim_unsure(t, p) + (box,) for t, p, box in zip(todo, probs, dbg["boxes"])]
        kept = [k for k in kept if k[0]]
        todo, dbg["boxes"] = [t for t, _, _ in kept], [box for _, _, box in kept]
        left = [q for _, p, _ in kept for q in p]                  # уверенность — по тому, что осталось
        confidence = float(np.mean(left)) if left else 0.0
    todo = [tidy(c) for c in todo]
    if not all(todo):                                             # от столбца ничего не осталось — и строки нет
        dbg["boxes"] = [b for t, b in zip(todo, dbg["boxes"]) if t]
        todo = [t for t in todo if t]
    letters = sum(ch not in _NOT_LETTERS for t in todo for ch in t)
    return todo, confidence, letters, dbg


def _better_turn(r, best) -> bool:
    """Повёрнутое прочтение r лучше прямого best: уверенно и прочитано заметно больше — или столько же,
    но заметно увереннее (короткий текст вверх ногами: 0.92 -> 0.98 при тех же буквах)."""
    if r is None or r[1] < ROTATED_MIN_CONFIDENCE or r[2] < FEW_LETTERS:   # пустой лист в любом повороте
        return False                                                       # даёт 3–4 уверенные «буквы»
    if best is None or _mass(r) > ROTATION_GAIN * _mass(best):
        return True
    return r[1] >= best[1] + 0.03 and _mass(r) >= 0.9 * _mass(best)


def _mass(r) -> float:
    """Сколько уверенно прочитано: число букв × средняя уверенность (для выбора поворота)."""
    return 0.0 if r is None else r[2] * r[1]


def _rank(r):
    """Какое прочтение лучше: сначала то, где есть буквы, потом — где модель увереннее."""
    return (-1, 0.0) if r is None else (int(r[2] >= MIN_LETTERS), r[1])


def warmup() -> dict:
    """Загрузить модель и прогнать пустой столбец: первый запрос не ждёт."""
    _split_page()
    m = get_model()
    m.read([Image.new("L", (64, 400), 255)])
    try:                                        # частоты слов и языковая модель для склейки
        from .broken_words import warmup as _warm_words
        _warm_words()
    except Exception:
        log.exception("склейка разорванных слов не прогрелась")
    return {"model_path": m.path, "alphabet": len(m.chars)}


# ---------------------------------------------------------------- картинка

def open_image(data: bytes) -> Image.Image:
    """Байты -> картинка в цвете (L, RGB или RGBA). В серое её переводит
    page_layout.to_gray: там решается, какой цвет текст, а какой фон."""
    try:
        im = Image.open(io.BytesIO(data))
        # JPEG можно декодировать сразу уменьшенным — быстрее и меньше памяти
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))
        im = ImageOps.exif_transpose(im)        # фото с телефона, присланное файлом, бывает повёрнуто через EXIF
        if "A" in im.mode or "transparency" in im.info:
            im = im.convert("RGBA")
        elif im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
    except Exception as exc:
        raise OcrError("Не получилось открыть картинку. Пришлите фото или файл PNG/JPG.") from exc
    if max(im.size) > MAX_SIDE:
        k = MAX_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)   # Pillow ≥ 10.1: масштабируемый встроенный шрифт
    except TypeError:
        return ImageFont.load_default()


def draw_columns(page: Image.Image, boxes: list, unread: list = ()):
    """Выпрямленная страница + рамки столбцов с номерами -> (JPEG, размер).
    unread — рамки размытых столбцов: серые, с «?» вместо номера (в тексте их нет)."""
    im = page.convert("RGB")
    k = min(1.0, OVERLAY_SIDE / max(im.size))
    if k < 1.0:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    line = max(2, round(max(im.size) / 500))
    widths = [(x1 - x0) * k for x0, _, x1, _ in boxes]
    font = _font(max(12, min(40, round((np.median(widths) if widths else 30) * 0.45))))
    marks = [(str(n), _BOX_COLORS[(n - 1) % 2], b) for n, b in enumerate(boxes, 1)]
    marks += [("?", _UNREAD_COLOR, b) for b in unread]
    for label, color, (x0, y0, x1, y1) in marks:
        box = [round(x0 * k), round(y0 * k), round(x1 * k), round(y1 * k)]
        box = [max(0, box[0]), max(0, box[1]), min(im.width - 1, box[2]), min(im.height - 1, box[3])]
        draw.rectangle(box, outline=color, width=line)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        cx = (box[0] + box[2]) // 2
        ty = box[1] - th - 2 * line if box[1] - th - 2 * line >= 0 else box[1] + line
        draw.rectangle([cx - tw // 2 - line, ty - line, cx + tw // 2 + line, ty + th + line], fill=color)
        draw.text((cx - tw // 2, ty), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), im.size


# --------------------------------------------------------------- основное

def recognize(data: bytes, overlay: bool = True, punctuation: str = DEFAULT_PUNCT) -> OcrResult:
    """Байты картинки -> OcrResult. Синхронно и не быстро (~1 с на страницу).

    punctuation — знаки препинания в ответе (PUNCT_* из punctuation.py): по
    умолчанию убираются и те, что модель прочитала на странице."""
    started = time.perf_counter()
    steps = {}
    # OcrUnavailable — до того, как тратить время на картинку
    split_page = _split_page()
    model = get_model()

    t0 = time.perf_counter()
    img = open_image(data)
    steps["картинка"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    best = _read_page(img, split_page, model)
    plate_box = None
    steps["разметка и модель"] = (time.perf_counter() - t0) * 1000
    poor = best is None or best[2] < FEW_LETTERS or best[1] < LOW_CONFIDENCE
    if poor or best[3].get("flip_gain", 0.0) >= FLIP_GAIN:
        # Фото или скан повёрнуты: у пустотной рукописи обороты листов сняты вверх ногами, титульная
        # строка — поперёк кадра. Вверх ногами модель читает почти уверенно (0.92 на синтетике), поэтому
        # переворот проверяется отдельно и дёшево — по самым длинным столбцам (flip_gain в _read_page); на
        # бок повёрнутая страница читается плохо сама (почти без букв), и тогда пробуем все три поворота.
        t0 = time.perf_counter()
        upright = img
        for rot in ((180, 90, 270) if poor else (180,)):
            turned = upright.rotate(rot, expand=True)
            r = _read_page(turned, split_page, model)
            if _better_turn(r, best):
                best, img = r, turned
        steps["поворот"] = (time.perf_counter() - t0) * 1000
    if best is None or best[2] < MIN_LETTERS or best[1] < LOW_CONFIDENCE:
        # Не страница во весь кадр, а надпись на предмете (кулон, табличка) на пёстром фоне:
        # ищем табличку и читаем её отдельно; берём, что прочиталось лучше.
        t0 = time.perf_counter()
        from .page_layout import find_plates
        for plate, box in find_plates(img):
            r = _read_page(plate, split_page, model, trim=True)
            if r is not None and r[2] < PLATE_MIN_LETTERS:
                continue
            if _rank(r) > _rank(best):
                best, plate_box = r, (box, plate.size)
        steps["поиск таблички"] = (time.perf_counter() - t0) * 1000
    if best is None:
        raise OcrError(
            "Не нашёл на картинке вертикального текста. Пришлите фото, где "
            "столбцы тодо бичик идут сверху вниз."
        )
    todo, confidence, letters, dbg = best
    if letters < MIN_LETTERS:
        raise OcrError(
            "Не разобрал на картинке тодо бичик. Нужно фото, где столбцы идут "
            "сверху вниз, а буквы крупные и чёткие."
        )

    t0 = time.perf_counter()
    # Страница размечена целиком — столбцы идут одним текстом: бирга перед
    # первым, четыре точки после последнего. Число строк не меняется, иначе
    # сбилась бы нумерация рамок.
    todo = apply_punctuation(normalize_ocr("\n".join(todo)), punctuation).split("\n")
    translit = [to_translit(t) for t in todo]
    steps["транслитерация"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cyrillic = to_cyrillic_columns(translit)
    steps["кириллица"] = (time.perf_counter() - t0) * 1000

    res = OcrResult(columns=todo, translit_columns=translit, cyrillic_columns=cyrillic,
                    confidence=confidence, angle=dbg["angle"], steps_ms=steps,
                    unreadable=0 if plate_box is not None else len(dbg.get("unread", [])))
    if overlay:
        t0 = time.perf_counter()
        if plate_box is not None:
            # рамки — на присланном фото: столбцы из выпрямленной вырезки переводим
            # обратно (поворот вокруг центра, сдвиг вырезки)
            (px0, py0, _, _), (pw, ph) = plate_box
            gh, gw = dbg["gray"].shape
            t = math.radians(dbg["angle"])
            c, s_ = math.cos(t), math.sin(t)

            def back(x, y):
                x, y = x - gw / 2, y - gh / 2
                return px0 + x * c - y * s_ + pw / 2, py0 + x * s_ + y * c + ph / 2

            boxes = []
            for x0, y0, x1, y1 in dbg["boxes"]:
                pts = [back(x, y) for x in (x0, x1) for y in (y0, y1)]
                boxes.append((min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)))
            res.overlay, res.overlay_size = draw_columns(img, boxes)
        elif img.mode == "RGBA":
            # у прозрачной картинки цвет под прозрачностью случайный — показываем то, что видела модель
            page = Image.fromarray(dbg["gray"])
            res.overlay, res.overlay_size = draw_columns(page, dbg["boxes"], dbg.get("unread", []))
        else:
            # рамки — на присланной картинке, в её цветах: так человек узнаёт своё фото
            page = img if not dbg["angle"] else img.rotate(
                dbg["angle"], Image.BICUBIC, expand=True,
                fillcolor=255 if img.mode == "L" else (255, 255, 255))
            res.overlay, res.overlay_size = draw_columns(page, dbg["boxes"], dbg.get("unread", []))
        steps["рамки"] = (time.perf_counter() - t0) * 1000
    res.elapsed_ms = (time.perf_counter() - started) * 1000
    return res
