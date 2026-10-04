# -*- coding: utf-8 -*-
"""Клавиатуры: постоянное меню режимов внизу и inline-кнопки под ответом.

payload кнопок (короткие ключи — у VK лимит 255 байт):
  {"cmd": "<команда>"}                  — кнопки нижнего меню (= /команда)
  {"a": "fb",   "v": "up"|"down", "r": <request_id>}   — оценка
  {"a": "conv", "t": "<режим>",   "r": <request_id>}   — «то же в другой записи»
  {"a": "set",  "do": "pick",   "f": <поле>, "p": <страница>} — открыть выбор
  {"a": "set",  "do": "set",    "f": <поле>, "v": <значение>}
  {"a": "set",  "do": "toggle", "f": "fix_letters"|"show_time"}
  {"a": "set",  "do": "back"} / {"a": "set", "do": "reset"}

Inline-кнопки — callback, если клиент их умеет: тогда нажатие не пишет в
чат лишних сообщений и экран настроек правится на месте. Иначе — те же
кнопки текстовыми (нажатие придёт сообщением с тем же payload). Клиенту
без inline-клавиатур кнопки под ответом не шлются вовсе.
"""

from typing import Optional

from core.todo_image import BG_PALETTE, PALETTE, color_label

from . import texts
from .vk import keyboard as kb
from .vk.events import ClientInfo

A_FEEDBACK = "fb"
A_CONVERT = "conv"
A_SETTINGS = "set"

# цветов на одной странице палитры: inline-клавиатура VK — до 10 кнопок,
# две уходят на «Назад» и «Ещё»
COLORS_PER_PAGE = 8


def _kind(client: ClientInfo) -> str:
    return kb.CALLBACK if client.callback else kb.TEXT


def main_menu(client: ClientInfo, mode: Optional[str] = None) -> Optional[str]:
    """Нижнее меню. Кнопка текущего режима — синяя."""
    if not client.keyboard:
        return None

    def mode_btn(key: str, label: str) -> dict:
        return kb.button(
            label, {"cmd": key}, kind=kb.TEXT,
            color="primary" if key == mode else "secondary",
        )

    return kb.keyboard([
        [mode_btn("translit", texts.BTN_TRANSLIT), mode_btn("todo", texts.BTN_TODO)],
        [mode_btn("image", texts.BTN_IMAGE), mode_btn("fix", texts.BTN_FIX)],
        [kb.button(texts.BTN_OCR, {"cmd": "ocr"}, kind=kb.TEXT),
         kb.button(texts.BTN_SETTINGS, {"cmd": "settings"}, kind=kb.TEXT)],
        [kb.button(texts.BTN_HELP, {"cmd": "help"}, kind=kb.TEXT)],
    ])


def result_keyboard(
    client: ClientInfo, request_id: int, target: str, with_feedback: bool = True
) -> Optional[str]:
    """Кнопки под результатом: продолжить конвертацию + оценка."""
    if not client.inline_keyboard:
        return None
    kind = _kind(client)

    def conv(label: str, to: str) -> dict:
        return kb.button(label, {"a": A_CONVERT, "t": to, "r": request_id}, kind=kind)

    convert = []
    if target == "fix":
        # исправленный текст — сразу дальше, в любой из трёх записей
        convert += [conv(texts.BTN_TO_TRANSLIT, "translit"), conv(texts.BTN_TO_TODO, "todo")]
    if target == "translit":
        convert.append(conv(texts.BTN_TO_TODO, "todo"))
    if target in ("translit", "todo", "fix"):
        convert.append(conv(texts.BTN_TO_IMAGE, "image"))

    rows = [convert]
    if with_feedback:
        rows.append([
            kb.button("👍", {"a": A_FEEDBACK, "v": "up", "r": request_id},
                      kind=kind, color="positive"),
            kb.button("👎", {"a": A_FEEDBACK, "v": "down", "r": request_id},
                      kind=kind, color="negative"),
        ])
    if not any(rows):
        return None
    return kb.keyboard(rows, inline=True)


def _set(kind: str, label: str, **payload) -> dict:
    return kb.button(label, {"a": A_SETTINGS, **payload}, kind=kind)


def _back(kind: str) -> dict:
    return _set(kind, texts.BTN_BACK, do="back")


def settings_menu(
    client: ClientInfo, opts, fix_letters: bool = False,
    punctuation: str = "off", show_time: bool = True,
) -> Optional[str]:
    """Главный экран настроек. Восемь кнопок в шесть рядов — больше
    рядов inline-клавиатура VK не берёт, поэтому картинка — по две в ряд."""
    if not client.inline_keyboard:
        return None
    k = _kind(client)
    return kb.keyboard([
        [_set(k, texts.SET_BTN_FG.format(value=color_label(opts.fg)), do="pick", f="fg"),
         _set(k, texts.SET_BTN_BG.format(value=color_label(opts.bg)), do="pick", f="bg")],
        [_set(k, texts.SET_BTN_SIZE.format(value=texts.SIZE_LABELS.get(opts.size, opts.size)),
              do="pick", f="size"),
         _set(k, texts.SET_BTN_FONT.format(value=texts.FONT_LABELS.get(opts.font, opts.font)),
              do="pick", f="font")],
        [_set(k, texts.SET_BTN_PUNCT.format(
            value=texts.PUNCT_LABELS.get(punctuation, punctuation)), do="pick", f="punctuation")],
        [_set(k, texts.SET_BTN_FIX.format(
            value=texts.FIX_LETTERS_ON if fix_letters else texts.FIX_LETTERS_OFF),
            do="toggle", f="fix_letters")],
        [_set(k, texts.SET_BTN_TIME.format(
            value=texts.SHOW_TIME_ON if show_time else texts.SHOW_TIME_OFF),
            do="toggle", f="show_time")],
        [_set(k, texts.SET_BTN_RESET, do="reset")],
    ], inline=True)


def _grid(buttons, per_row=2):
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def _mark(label: str, chosen: bool) -> str:
    return label + (" ✓" if chosen else "")


def color_picker(client: ClientInfo, field: str, current: str, page: int = 0) -> str:
    """Палитра. Текущий цвет помечен галочкой. Цветов больше, чем влезает
    в одну inline-клавиатуру VK, поэтому палитра листается."""
    k = _kind(client)
    palette = BG_PALETTE if field == "bg" else PALETTE
    pages = max(1, -(-len(palette) // COLORS_PER_PAGE))
    page = min(max(page, 0), pages - 1)
    chunk = palette[page * COLORS_PER_PAGE : (page + 1) * COLORS_PER_PAGE]
    buttons = [
        _set(k, _mark(f"{chip} {label}", key == current), do="set", f=field, v=key)
        for key, label, chip in chunk
    ]
    nav = [_back(k)]
    if pages > 1:
        if page + 1 < pages:
            nav.append(_set(k, texts.BTN_MORE, do="pick", f=field, p=page + 1))
        else:
            nav.append(_set(k, texts.BTN_LESS, do="pick", f=field, p=page - 1))
    return kb.keyboard(_grid(buttons) + [nav], inline=True)


def size_picker(client: ClientInfo, current: str) -> str:
    k = _kind(client)
    buttons = [
        _set(k, _mark(label, key == current), do="set", f="size", v=key)
        for key, label in texts.SIZE_LABELS.items()
    ]
    return kb.keyboard([buttons, [_back(k)]], inline=True)


def font_picker(client: ClientInfo, current: str) -> str:
    k = _kind(client)
    buttons = [
        _set(k, _mark(label, key == current), do="set", f="font", v=key)
        for key, label in texts.FONT_LABELS.items()
    ]
    return kb.keyboard(_grid(buttons) + [[_back(k)]], inline=True)


def punct_picker(client: ClientInfo, current: str) -> str:
    """Три режима знаков препинания — каждый своей строкой: подписи длинные."""
    k = _kind(client)
    rows = [
        [_set(k, _mark(label, key == current), do="set", f="punctuation", v=key)]
        for key, label in texts.PUNCT_LABELS.items()
    ]
    return kb.keyboard(rows + [[_back(k)]], inline=True)
