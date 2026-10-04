# -*- coding: utf-8 -*-
"""
todo_image.py — рендер транслитерации (или готового текста тодо бичиг) в
картинку с настоящим вертикальным письмом.

Как устроено вертикальное письмо:
  1. Каждая "логическая строка" рисуется ГОРИЗОНТАЛЬНО как обычно (шрифт
     не поддерживает вертикальный режим сам по себе).
  2. Картинка строки поворачивается на 90° ПО ЧАСОВОЙ стрелке
     (`rotate(-90)` в PIL): левый край становится верхним, чтение
     слева-направо превращается в чтение сверху-вниз.
  3. Если строка не помещается по высоте (после поворота — это высота
     столбца) — переносим по словам на несколько под-строк/столбцов.
  4. Получившиеся столбцы выстраиваются СЛЕВА НАПРАВО.

Перенос — только по пробелу. Символ границы суффикса (узкий
неразрывный пробел, \\u202f) переносить нельзя.

Отличия от исходного скрипта: путь к шрифту берётся из переменной
окружения FONT_PATH (по умолчанию assets/MongolianUniversalWhite.ttf),
добавлены поля вокруг картинки и функция render_todo_bytes(), которая
отдаёт PNG в память — так его удобно сразу слать в Telegram.
"""

import io
import os
import warnings

from PIL import Image, ImageDraw, ImageFont, features

from . import shaper
from .translit_todo import translit_to_todo

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FONT = os.path.join(
    os.path.dirname(_HERE), "assets", "MongolianUniversalWhite.ttf"
)
DEFAULT_TODO_FONT = os.environ.get("FONT_PATH", _DEFAULT_FONT)

_NNBSP = " "


# =============================================================================
# ГЛАВНОЕ ТРЕБОВАНИЕ К ОКРУЖЕНИЮ: Pillow, собранный с Raqm
# =============================================================================
#
# Монгольское письмо (и тодо бичиг вместе с ним) — курсивное: буква имеет
# разные формы в начале, середине и конце слова, и соседние буквы сливаются
# в один вертикальный стержень. Выбор нужной формы делает не шрифт сам по
# себе, а движок раскладки текста — HarfBuzz, к которому Pillow ходит через
# библиотеку Raqm.
#
# Если Raqm в сборке Pillow нет, Pillow МОЛЧА откатывается на примитивную
# раскладку: каждая буква рисуется в изолированной форме, буквы не
# соединяются. Картинка получается похожей на текст, но читать её трудно.
# Есть Raqm в конкретной сборке или нет — свойство окружения: в Colab и в
# колёсах с PyPI он есть, в conda и системных пакетах часто нет.
#
# Поэтому основной путь здесь — НЕ Pillow: модуль shaper.py зовёт HarfBuzz
# напрямую (uharfbuzz) и растеризует глифы через FreeType. Работает
# одинаково в любом окружении и от сборки Pillow не зависит вовсе.
#
# Порядок выбора движка:
#   1. uharfbuzz + freetype-py  — основной, ставится из requirements.txt;
#   2. Pillow с Raqm            — если первых нет, но повезло со сборкой;
#   3. Pillow без Raqm          — рисует несоединённые буквы, бот помечает
#                                 такую картинку предупреждением.
# STRICT_SHAPING=1 запрещает третий вариант вместо предупреждения.

HAS_RAQM = features.check("raqm")
HAS_HARFBUZZ = shaper.HAS_HARFBUZZ

_LAYOUT = (
    ImageFont.Layout.RAQM
    if HAS_RAQM and hasattr(ImageFont, "Layout")
    else getattr(getattr(ImageFont, "Layout", None), "BASIC", None)
)

STRICT_SHAPING = os.environ.get("STRICT_SHAPING", "").lower() in {
    "1", "true", "yes", "on", "да",
}

# буквы соединяются — то есть письмо настоящее, а не набор изолированных форм
SHAPING_OK = HAS_HARFBUZZ or HAS_RAQM
SHAPING_ENGINE = "harfbuzz" if HAS_HARFBUZZ else ("raqm" if HAS_RAQM else "none")

_NO_SHAPING_MESSAGE = (
    "Буквы тодо бичиг не соединяются: нет ни uharfbuzz, ни Pillow с Raqm. "
    "Поставьте зависимости проекта: pip install -r requirements.txt "
    "(нужны uharfbuzz и freetype-py) — это лечит проблему в любом окружении."
)

# короткая версия — её бот показывает пользователю под картинкой.
# Про Pillow, Raqm и логи человеку в чате знать незачем: это забота того,
# кто бота развернул, ему подробности уходят в лог при старте.
SHAPING_WARNING = (
    "⚠️ Буквы на картинке не соединены — на сервере не хватает библиотеки. "
    "Напишите администратору бота."
)


class ShapingUnavailable(RuntimeError):
    """Соединять буквы нечем, а STRICT_SHAPING=1 требует этого."""


def require_shaping() -> None:
    """Проверка перед рендером. Ругается только в строгом режиме — иначе
    рендер продолжается, а предупреждение уходит пользователю."""
    if SHAPING_OK or not STRICT_SHAPING:
        return
    raise ShapingUnavailable(_NO_SHAPING_MESSAGE)


def shaping_status() -> str:
    if HAS_HARFBUZZ:
        return f"буквы соединяются, движок: HarfBuzz напрямую ({shaper.status()})"
    if HAS_RAQM:
        return (
            "буквы соединяются, движок: Pillow + Raqm "
            f"(harfbuzz {features.version('harfbuzz')}). "
            "Поставьте uharfbuzz, чтобы не зависеть от сборки Pillow"
        )
    if STRICT_SHAPING:
        return "движка раскладки НЕТ, STRICT_SHAPING=1 — рендер картинок отключён"
    return "движка раскладки НЕТ — картинки рисуются, но буквы не соединяются"


# =============================================================================
# Цвета — по словам, RU и EN, с прозрачным фоном отдельным вариантом.
# =============================================================================

BASE_COLORS = {
    "black": (0, 0, 0, 255), "чёрный": (0, 0, 0, 255), "черный": (0, 0, 0, 255),
    "white": (255, 255, 255, 255), "белый": (255, 255, 255, 255),
    "red": (214, 39, 40, 255), "красный": (214, 39, 40, 255),
    "green": (44, 160, 44, 255), "зелёный": (44, 160, 44, 255), "зеленый": (44, 160, 44, 255),
    "blue": (31, 119, 180, 255), "синий": (31, 119, 180, 255),
    "yellow": (230, 200, 0, 255), "жёлтый": (230, 200, 0, 255), "желтый": (230, 200, 0, 255),
    "orange": (255, 127, 14, 255), "оранжевый": (255, 127, 14, 255),
    "purple": (148, 103, 189, 255), "фиолетовый": (148, 103, 189, 255),
    "gray": (128, 128, 128, 255), "grey": (128, 128, 128, 255), "серый": (128, 128, 128, 255),
    "brown": (140, 86, 75, 255), "коричневый": (140, 86, 75, 255),
    "pink": (227, 119, 194, 255), "розовый": (227, 119, 194, 255),
    "cyan": (23, 190, 207, 255), "голубой": (23, 190, 207, 255),
    "gold": (212, 175, 55, 255), "золотой": (212, 175, 55, 255),
    "cream": (247, 242, 230, 255), "кремовый": (247, 242, 230, 255),
}

_TRANSPARENT_NAMES = {"transparent", "прозрачный", "none", "нет"}

DEFAULT_FG = BASE_COLORS["black"]
DEFAULT_BG = BASE_COLORS["white"]

TRANSPARENT = "transparent"

# Палитра для кнопок в боте: (ключ, подпись, квадратик).
# Держим её здесь, рядом с самими цветами, чтобы телеграм-слой не заводил
# собственный список и они не разъехались. Эмодзи — только те, что рисуются
# и на старых клиентах: цветные квадраты из Unicode 9/12.
PALETTE = [
    ("black", "чёрный", "⬛"),
    ("white", "белый", "⬜"),
    ("red", "красный", "🟥"),
    ("orange", "оранжевый", "🟧"),
    ("yellow", "жёлтый", "🟨"),
    ("green", "зелёный", "🟩"),
    ("blue", "синий", "🟦"),
    ("purple", "фиолетовый", "🟪"),
    ("brown", "коричневый", "🟫"),
    ("gold", "золотой", "🟡"),
]

# фон умеет ещё и прозрачность — для наложения на другие картинки
BG_PALETTE = PALETTE + [(TRANSPARENT, "прозрачный", "▫️")]

_PALETTE_LABELS = {key: (label, chip) for key, label, chip in BG_PALETTE}


def color_label(key: str) -> str:
    """«red» -> «🟥 красный». Для показа текущих настроек."""
    label, chip = _PALETTE_LABELS.get(key, ("", ""))
    return f"{chip} {label}".strip() or str(key)


def is_transparent(name) -> bool:
    return isinstance(name, str) and name.strip().lower() in _TRANSPARENT_NAMES


def resolve_color(name, fallback, allow_transparent=False):
    """Название цвета (по-русски или по-английски) -> RGBA. Если имя не
    из базового набора — берём fallback."""
    if name is None:
        return fallback
    if not isinstance(name, str):
        return name  # уже готовый tuple/цвет — пропускаем как есть
    key = name.strip().lower()
    if allow_transparent and key in _TRANSPARENT_NAMES:
        return (0, 0, 0, 0)
    if key in BASE_COLORS:
        return BASE_COLORS[key]
    warnings.warn(f"цвет «{name}» не из базового набора, беру цвет по умолчанию")
    return fallback


def _resolve_fg_bg(fg, bg):
    """Возвращает (цвет текста, цвет фона, был ли откат).

    Откат нужен, когда цвета совпали: текст на таком фоне попросту не
    видно, и честнее отдать читаемую картинку чёрным по белому, чем
    пустой прямоугольник. Третьим значением сообщаем об этом наружу,
    чтобы бот мог предупредить — иначе человек решит, что бот сломался."""
    fg_rgba = resolve_color(fg, DEFAULT_FG)
    bg_rgba = resolve_color(bg, DEFAULT_BG, allow_transparent=True)
    if fg_rgba == bg_rgba:
        return DEFAULT_FG, DEFAULT_BG, True
    return fg_rgba, bg_rgba, False


def _load_font(font_path, font_size):
    if font_path is None:
        try:
            return ImageFont.load_default(size=font_size)
        except TypeError:
            # старые версии Pillow не умеют load_default(size=...)
            return ImageFont.load_default()
    # layout_engine указываем явно: без него Pillow сам решает, чем рисовать,
    # и при отсутствии Raqm тихо берёт примитивную раскладку
    if _LAYOUT is not None:
        return ImageFont.truetype(font_path, font_size, layout_engine=_LAYOUT)
    return ImageFont.truetype(font_path, font_size)


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox


# --- две реализации одного и того же: измерить строку и нарисовать строку ---
# Выше по коду выбирается, какая из них в ходу. Вызывающий код (перенос по
# словам, сборка столбцов) от выбора не зависит.


def _use_harfbuzz(font_path):
    # шрифт по умолчанию от PIL — не файл, HarfBuzz его не откроет
    return HAS_HARFBUZZ and font_path is not None


def _measure_width(text, font, font_path, font_size, draw_probe):
    if _use_harfbuzz(font_path):
        return shaper.measure(text, font_path, font_size)
    return _text_size(draw_probe, text, font)[0]


def _render_strip(text, font, font_path, font_size, fg, bg, pad=6):
    """Строка, нарисованная горизонтально (поворот — снаружи)."""
    if _use_harfbuzz(font_path):
        return shaper.render_line(text, font_path, font_size, fg, bg, pad=pad)
    w, h, bbox = _text_size(ImageDraw.Draw(Image.new("RGBA", (10, 10))), text, font)
    strip = Image.new("RGBA", (w + pad * 2, h + pad * 2), bg)
    ImageDraw.Draw(strip).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=fg
    )
    return strip


def _wrap_line(line, font, font_path, font_size, max_column_height, draw):
    """Разбивает одну строку тодо-бичиг текста на под-строки так, чтобы
    каждая под-строка при горизонтальном рендере не превышала по ширине
    max_column_height (это и есть будущая высота столбца после поворота).
    Перенос — по обычному пробелу; \\u202f внутри "слова" не трогаем.
    Пробелы по краям и двойные пробелы пустых «слов» не дают: строка
    « ︱ сән» (тире в начале реплики) иначе начиналась бы с пустого
    кусочка, а пустую строку HarfBuzz не раскладывает."""
    words = [word for word in line.split(" ") if word]
    sublines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        w = _measure_width(candidate, font, font_path, font_size, draw)
        if w <= max_column_height or not current:
            current = candidate
        else:
            sublines.append(current)
            current = word
    if current:
        sublines.append(current)
    return sublines


def _render_column(subline, font, font_path, font_size, fg, bg, pad=6):
    """Рисует одну под-строку горизонтально, затем поворачивает на 90°
    по часовой — получается один готовый вертикальный столбец."""
    strip = _render_strip(subline, font, font_path, font_size, fg, bg, pad=pad)
    return strip.rotate(-90, expand=True)


def _build_columns(
    lines,
    font_size=64,
    max_column_height=900,
    fg="black",
    bg="white",
    font_path=DEFAULT_TODO_FONT,
):
    """Текст -> готовые вертикальные столбцы.

    Возвращает (столбцы, цвет текста, цвет фона, был ли откат по цветам).
    Каждый столбец — (картинка, начинает ли он новую логическую строку):
    по второму значению собиратель понимает, какой отступ ставить слева.
    """
    require_shaping()

    if isinstance(lines, str):
        lines = lines.split("\n")

    fg, bg, color_fallback = _resolve_fg_bg(fg, bg)
    font = _load_font(font_path, font_size)
    draw_probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

    columns = []
    for line in lines:
        line = shaper.fit_to_font(line.strip("\n"), font_path)
        if not line.strip():
            continue
        sublines = _wrap_line(
            line, font, font_path, font_size, max_column_height, draw_probe
        )
        for i, subline in enumerate(sublines):
            col = _render_column(subline, font, font_path, font_size, fg, bg)
            columns.append((col, i == 0))

    if not columns:
        raise ValueError("нечего рендерить — пустой текст")
    return columns, fg, bg, color_fallback


def _gap_before(index, is_new_line, column_gap, line_gap):
    """Отступ слева от столбца. У первого на листе отступа нет."""
    if index == 0:
        return 0
    return line_gap if is_new_line else column_gap


def _page_width(columns, column_gap, line_gap, margin):
    width = margin * 2
    for i, (col, is_new_line) in enumerate(columns):
        width += _gap_before(i, is_new_line, column_gap, line_gap) + col.width
    return width


def _compose(columns, bg, column_gap, line_gap, margin, height=None):
    """Столбцы -> один лист. Столбцы идут слева направо.

    height задаётся снаружи, когда листов несколько: высота столбца зависит
    от того, насколько длинной вышла последняя строка на листе, и без общей
    высоты листы получались бы чуть разными — в чате это сразу видно."""
    width = _page_width(columns, column_gap, line_gap, margin)
    height = height or max(c.height for c, _ in columns) + margin * 2
    canvas = Image.new("RGBA", (width, height), bg)
    x = margin
    for i, (col, is_new_line) in enumerate(columns):
        x += _gap_before(i, is_new_line, column_gap, line_gap)
        canvas.paste(col, (x, margin), col)
        x += col.width
    return canvas


def _split_pages(columns, max_width, column_gap, line_gap, margin):
    """Режет поток столбцов на листы по ширине.

    Порядок столбцов не меняется, поэтому листы читаются подряд: первый,
    второй и так далее. Один столбец шире листа влезть не может по
    построению (его ширина — это кегль), так что бесконечного цикла нет.
    """
    pages, current, width = [], [], margin * 2
    for col, is_new_line in columns:
        need = _gap_before(len(current), is_new_line, column_gap, line_gap) + col.width
        if current and width + need > max_width:
            pages.append(current)
            current, width = [], margin * 2
            need = col.width
        current.append((col, is_new_line))
        width += need
    if current:
        pages.append(current)
    return pages


def render_todo_paragraph(
    lines,
    out_path=None,
    font_size=64,
    max_column_height=900,
    column_gap=14,
    line_gap=40,
    margin=36,
    fg="black",
    bg="white",
    font_path=DEFAULT_TODO_FONT,
):
    """
    lines    — текст УЖЕ в юникоде тодо бичиг: одна строка (можно с \\n
               внутри) или список строк.
    fg, bg   — название цвета словом (см. BASE_COLORS), не hex-код.
               bg="transparent"/"прозрачный" — пустой фон.
    out_path — если задан, картинка ещё и сохраняется на диск.
    Возвращает объект PIL.Image; атрибут .color_fallback на нём говорит,
    пришлось ли откатиться на чёрное-по-белому из-за совпавших цветов.
    """
    columns, _fg, bg, color_fallback = _build_columns(
        lines, font_size=font_size, max_column_height=max_column_height,
        fg=fg, bg=bg, font_path=font_path,
    )
    canvas = _compose(columns, bg, column_gap, line_gap, margin)

    if out_path:
        canvas.save(out_path)
    # PIL.Image — обычный объект, поле на нём переживёт resize только если
    # его перенести руками (см. render_todo_bytes)
    canvas.color_fallback = color_fallback
    return canvas


def render_todo_image(text, out_path=None, **kwargs):
    """
    text — транслитерация проекта (не тодо бичиг, не кириллица). Можно
    несколько строк через \\n или списком — конвертирует каждую и зовёт
    render_todo_paragraph.
    """
    lines = text.split("\n") if isinstance(text, str) else text
    todo_lines = [translit_to_todo(line) for line in lines]
    return render_todo_paragraph(todo_lines, out_path=out_path, **kwargs)


MAX_SIDE = 2600  # у Telegram сумма сторон фото ограничена, да и смысла нет

# Сколько листов согласны отправить за один запрос. Каждый лист — отдельное
# сообщение, а Telegram считает их частоту, поэтому счёт идёт на десятки, а
# не на сотни. Двадцати хватает на десять тысяч символов (MAX_INPUT_CHARS)
# даже крупным кеглем: там выходит шестнадцать листов.
MAX_PAGES = 20

# К чему стремимся по большей стороне. Шрифт векторный, «разрешения» у него
# нет — сколько пикселей попросим, столько и нарисует. Но кегль задаёт ещё и
# перенос по столбцам, поэтому просто поднять его нельзя: изменится вёрстка.
# Поэтому кегль и высоту столбца умножаем на один и тот же множитель —
# картинка та же, только плотнее. Без этого одно слово давало 133x215, и на
# экране, где его растягивают, была видна лестница по краям букв.
TARGET_MIN_SIDE = 1200
MAX_SCALE = 6  # выше смысла нет: упрёмся в MAX_SIDE


def _quality_scale(size):
    """Во сколько раз перерисовать, чтобы картинка перестала быть мелкой.

    Множитель дробный намеренно: при округлении вниз до целого картинки
    размером от 600 до 1200 px не получали ничего (1200/654 -> 1), то есть
    ровно средние случаи оставались мелкими."""
    longest = max(size)
    if longest >= TARGET_MIN_SIDE:
        return 1.0
    return min(MAX_SCALE, TARGET_MIN_SIDE / longest)


def _to_png(img, max_side):
    """Картинка -> PNG в памяти. Слишком большую ужимаем по большей стороне."""
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf, img.size


def render_todo_pages(todo_text, max_side=MAX_SIDE, max_pages=MAX_PAGES, **kwargs):
    """Готовый текст тодо бичиг -> список листов-PNG. Этим пользуется бот.

    Длинный текст в одну картинку не влезает: столбцы идут слева направо, и
    чтобы вместить их все, картинку пришлось бы ужать до нечитаемого.
    Поэтому поток столбцов режется на листы шириной не больше max_side —
    каждый лист остаётся крупным, а читаются они подряд.

    Возвращает (листы, meta), где лист — (BytesIO, (ширина, высота)), а
    meta — словарь с color_fallback (пришлось ли спасать совпавшие цвета),
    transparent (прозрачный ли фон; такую картинку нельзя слать как фото —
    Telegram пережмёт её в JPEG и прозрачность станет чёрным) и truncated
    (сколько листов не поместилось в max_pages).
    """
    column_gap = kwargs.get("column_gap", 14)
    line_gap = kwargs.get("line_gap", 40)
    margin = kwargs.get("margin", 36)
    build_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ("column_gap", "line_gap", "margin")}

    columns, _fg, bg, color_fallback = _build_columns(todo_text, **build_kwargs)

    # Мелкую картинку перерисовываем крупнее — именно перерисовываем, а не
    # растягиваем: растягивание добавит размытия, но не деталей. Длинного
    # текста это не касается, он и так во весь лист.
    width = _page_width(columns, column_gap, line_gap, margin)
    height = max(c.height for c, _ in columns) + margin * 2
    scale = _quality_scale((width, height)) if width <= max_side else 1.0
    if scale > 1.01:
        # все величины умножаем на один множитель — вёрстка остаётся прежней
        # до пикселя, меняется только плотность
        margin, column_gap, line_gap = (round(v * scale)
                                        for v in (margin, column_gap, line_gap))
        columns, _fg, bg, color_fallback = _build_columns(todo_text, **{
            **build_kwargs,
            "font_size": round(kwargs.get("font_size", 64) * scale),
            "max_column_height": round(
                kwargs.get("max_column_height", 900) * scale),
        })

    sheets = _split_pages(columns, max_side, column_gap, line_gap, margin)
    # высота у всех листов общая — по самому длинному столбцу
    height = max(c.height for c, _ in columns) + margin * 2
    pages = [
        _to_png(
            _compose(sheet, bg, column_gap, line_gap, margin, height), max_side
        )
        for sheet in sheets[:max_pages]
    ]
    meta = {
        "color_fallback": color_fallback,
        # прозрачность запрашивали и её не отменил откат по совпавшим цветам
        "transparent": is_transparent(kwargs.get("bg")) and not color_fallback,
        "truncated": max(0, len(sheets) - max_pages),
    }
    return pages, meta


def render_todo_bytes(todo_text, max_side=MAX_SIDE, **kwargs):
    """Первый лист — для кода, которому хватает одной картинки."""
    pages, meta = render_todo_pages(todo_text, max_side=max_side, **kwargs)
    buf, size = pages[0]
    return buf, size, meta

if __name__ == "__main__":
    print("Раскладка текста:", shaping_status())
    sentence = "xalimaq ulus ger-yēn naran üde xalīlγaǰi baridaq"
    render_todo_image(sentence, out_path="demo_sentence.png", max_column_height=260)
    print("сохранено: demo_sentence.png")

    paragraph = [
        "ene yertünci dü mini oron",
        "eberē amidural yin ecen bi",
        "külüg yin mini kacar-yēn tatači bayikuši bi",
        "küsel yēn dakāqd sanā bēn cingnēd",
    ]
    render_todo_paragraph(
        [translit_to_todo(line) for line in paragraph],
        out_path="demo_paragraph.png", max_column_height=500,
    )
    print("сохранено: demo_paragraph.png")
