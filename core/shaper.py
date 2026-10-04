# -*- coding: utf-8 -*-
"""
shaper.py — собственный рендер строки тодо бичиг: HarfBuzz для раскладки
плюс FreeType для растеризации. Без Pillow-овского рисования текста.

ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО

Монгольское письмо курсивное: у буквы разные формы в начале, середине и
конце слова, соседние буквы сливаются в один стержень. Нужную форму
выбирают таблицы GSUB/GPOS в шрифте, а применяет их движок раскладки —
HarfBuzz. Pillow умеет ходить в HarfBuzz, но только если собран с
библиотекой Raqm; если нет — МОЛЧА рисует изолированные формы, и текст
получается нечитаемым. Есть Raqm в конкретной сборке Pillow или нет —
свойство окружения, которое меняется от машины к машине (в Colab есть,
в conda часто нет).

Здесь мы убираем эту зависимость: зовём HarfBuzz напрямую через
uharfbuzz, получаем список глифов с позициями, и рисуем каждый глиф
растеризатором FreeType. Результат одинаковый везде, где ставятся
uharfbuzz и freetype-py, — от ноутбука до докера на хостинге.

КАК УСТРОЕНО

  1. hb.Buffer со строкой, direction="ltr", script="Mong".
     Направление именно горизонтальное: у этого шрифта нет вертикальных
     метрик (при direction="ttb" HarfBuzz выдаёт одинаковый шаг -2380 для
     всех глифов, то есть ничего осмысленного), а сами глифы нарисованы
     уже повёрнутыми под вертикальное письмо. Поэтому строка рисуется
     горизонтально и целиком поворачивается на 90° — так же, как это
     делалось раньше через Pillow.
  2. hb.shape() применяет GSUB/GPOS и возвращает глифы (уже в нужных
     позиционных формах) с офсетами и шагами в 1/64 пикселя.
  3. FreeType растеризует каждый глиф по его номеру; кладём получившиеся
     8-битные маски на общий холст со сдвигами от HarfBuzz.
  4. Маска используется как альфа: заливаем ей цвет текста поверх фона.
     Прозрачный фон при этом остаётся прозрачным.

Размеры и позиции HarfBuzz отдаёт в формате 26.6 (целое = 1/64 пикселя),
отсюда деления на 64 по всему файлу.
"""

import os
import unicodedata
from functools import lru_cache

from PIL import Image, ImageChops

try:
    import freetype
    import uharfbuzz as hb

    HAS_HARFBUZZ = True
    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    HAS_HARFBUZZ = False
    IMPORT_ERROR = exc

# HarfBuzz сам разберётся с языком, но подсказать скрипт стоит: строка
# может начинаться с пробела или цифры, и тогда автоопределение промахнётся
SCRIPT = "Mong"


class ShapedRun:
    """Результат раскладки одной строки: глифы, их позиции и габариты."""

    __slots__ = ("glyphs", "width", "ink_left", "ink_top", "ink_width", "ink_height")

    def __init__(self, glyphs, width, ink_left, ink_top, ink_width, ink_height):
        self.glyphs = glyphs  # [(gid, x_px, y_px, bitmap)]
        self.width = width  # ширина по шагам глифов (с учётом пробелов)
        self.ink_left = ink_left
        self.ink_top = ink_top
        self.ink_width = ink_width
        self.ink_height = ink_height


@lru_cache(maxsize=8)
def _load(font_path: str, size: int):
    """Открываем шрифт дважды: для HarfBuzz (раскладка) и для FreeType
    (растеризация). Кэшируем — файл шрифта полмегабайта."""
    with open(font_path, "rb") as fh:
        data = fh.read()
    face = hb.Face(hb.Blob(data))
    font = hb.Font(face)
    # scale в 26.6: позиции сразу приедут в 1/64 пикселя нужного кегля
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)

    ft = freetype.Face(font_path)
    ft.set_pixel_sizes(0, size)
    return font, ft


@lru_cache(maxsize=16)
def _charset(font_path: str) -> frozenset:
    """Какие символы вообще есть в шрифте."""
    return frozenset(code for code, _gid in freetype.Face(font_path).get_chars())


# Чего нет в шрифте — похожим знаком, который обычно есть. Замена берётся,
# только если сама она в шрифте нашлась.
_SUBSTITUTES = {
    "…": "᠁",
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "—": "︱", "–": "︱", "―": "︱",
    "„": "«", "︽": "«", "︾": "»",
    "\u00a0": " ", "\u202f": " ",
}


def fit_to_font(text: str, font_path: str) -> str:
    """Убирает из строки символы, которых в шрифте нет: иначе они выходят
    на картинке пустыми квадратиками (эмодзи, €, ×, … в монгольском
    шрифте). Сначала пробуем похожий знак, не вышло — выбрасываем.
    Невидимые управляющие символы (ZWJ, FVS) не трогаем: они влияют на
    формы букв, а не рисуются сами."""
    if not HAS_HARFBUZZ or not font_path:
        return text
    have = _charset(font_path)
    out = []
    for ch in text:
        if ch in " \n" or ord(ch) in have or unicodedata.category(ch) == "Cf":
            out.append(ch)
            continue
        sub = _SUBSTITUTES.get(ch)
        if sub and all(c == " " or ord(c) in have for c in sub):
            out.append(sub)
        elif not out or out[-1] != " ":
            # вместо выброшенного — пробел: «5€/кг» не должно слипнуться
            out.append(" ")
    return "".join(out)


@lru_cache(maxsize=4096)
def _glyph_bitmap(font_path: str, size: int, gid: int):
    """Растеризованный глиф: (маска 'L', left, top). Кэш нужен всерьёз —
    в калмыцком тексте одни и те же буквы идут по кругу."""
    _, ft = _load(font_path, size)
    ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    slot = ft.glyph
    bmp = slot.bitmap
    if bmp.width == 0 or bmp.rows == 0:
        return None, slot.bitmap_left, slot.bitmap_top

    # ВАЖНО: bmp.buffer в freetype-py — property, которая каждый раз
    # собирает новый python-список на rows*pitch элементов. Если трогать
    # её внутри цикла по строкам, получается квадратичная работа: на
    # мелком кегле незаметно, на крупном один глиф считается секундами.
    # Поэтому забираем буфер ровно один раз.
    data = bytes(bmp.buffer)
    pitch = abs(bmp.pitch)
    if pitch != bmp.width:
        # строки с выравниванием — вырезаем полезную часть каждой
        data = b"".join(
            data[r * pitch: r * pitch + bmp.width] for r in range(bmp.rows)
        )
    img = Image.frombytes("L", (bmp.width, bmp.rows), data)
    return img, slot.bitmap_left, slot.bitmap_top


def shape(text: str, font_path: str, size: int) -> ShapedRun:
    """Раскладка строки: какие глифы, где именно и какого размера итог."""
    if not HAS_HARFBUZZ:  # pragma: no cover
        raise RuntimeError(f"uharfbuzz/freetype-py не установлены: {IMPORT_ERROR}")

    if not text:
        # HarfBuzz на пустой строке отдаёт None вместо списков глифов
        return ShapedRun([], 0, 0, 0, 0, 0)

    font, _ = _load(font_path, size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = "ltr"
    buf.script = SCRIPT
    hb.shape(font, buf)

    glyphs = []
    pen_x = pen_y = 0  # в 1/64 пикселя
    ink_x0 = ink_y0 = 10**9
    ink_x1 = ink_y1 = -(10**9)

    for info, pos in zip(buf.glyph_infos or [], buf.glyph_positions or []):
        bitmap, left, top = _glyph_bitmap(font_path, size, info.codepoint)
        x = round((pen_x + pos.x_offset) / 64) + left
        # ось Y у FreeType вверх, у картинки вниз — отсюда минус
        y = round((pen_y + pos.y_offset) / 64) - top
        if bitmap is not None:
            glyphs.append((info.codepoint, x, y, bitmap))
            ink_x0 = min(ink_x0, x)
            ink_y0 = min(ink_y0, y)
            ink_x1 = max(ink_x1, x + bitmap.width)
            ink_y1 = max(ink_y1, y + bitmap.height)
        pen_x += pos.x_advance
        pen_y += pos.y_advance

    if not glyphs:
        # строка из одних пробелов — чернил нет, но ширина есть
        return ShapedRun([], round(pen_x / 64), 0, 0, 0, 0)

    return ShapedRun(
        glyphs,
        round(pen_x / 64),
        ink_x0,
        ink_y0,
        ink_x1 - ink_x0,
        ink_y1 - ink_y0,
    )


def measure(text: str, font_path: str, size: int) -> int:
    """Ширина строки в пикселях — по чернилам, как это делает Pillow
    в textbbox(). Нужно для переноса по словам."""
    run = shape(text, font_path, size)
    return run.ink_width or run.width


def render_line(text: str, font_path: str, size: int, fg, bg, pad: int = 6):
    """Одна строка, нарисованная ГОРИЗОНТАЛЬНО. Поворот на 90° делает
    вызывающий код — он же собирает столбцы."""
    run = shape(text, font_path, size)
    width = max(1, run.ink_width) + pad * 2
    height = max(1, run.ink_height) + pad * 2

    # собираем альфа-маску из глифов, потом заливаем её цветом
    mask = Image.new("L", (width, height), 0)
    for _gid, x, y, bitmap in run.glyphs:
        bx, by = x - run.ink_left + pad, y - run.ink_top + pad
        # в курсивном письме соседние глифы заходят друг на друга, поэтому
        # накладываем по максимуму, а не затираем прямоугольником: иначе
        # хвост предыдущей буквы срезало бы фоном следующей
        region = mask.crop((bx, by, bx + bitmap.width, by + bitmap.height))
        mask.paste(ImageChops.lighter(region, bitmap), (bx, by))

    canvas = Image.new("RGBA", (width, height), bg)
    canvas.paste(Image.new("RGBA", (width, height), fg), (0, 0), mask)
    return canvas


def status() -> str:
    if not HAS_HARFBUZZ:
        return f"uharfbuzz/freetype-py нет ({IMPORT_ERROR})"
    return (
        f"uharfbuzz {getattr(hb, '__version__', '?')}, "
        f"freetype-py {'.'.join(map(str, freetype.version()))}"
    )


if __name__ == "__main__":
    from .translit_todo import translit_to_todo

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "MongolianUniversalWhite.ttf",
    )
    print(status())
    run = shape(translit_to_todo("xalimaq ulus"), path, 64)
    print(f"глифов: {len(run.glyphs)}, чернила: {run.ink_width}x{run.ink_height}")
