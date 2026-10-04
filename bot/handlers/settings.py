# -*- coding: utf-8 -*-
"""
Настройки — цвет текста, цвет фона, размер и шрифт для картинки,
исправление текста, набранного без калмыцких букв (core/fix_letters.py;
по умолчанию выключено — правильно набранный текст не должен меняться
без спроса), знаки препинания тодо бичиг (core/punctuation.py; по
умолчанию не ставятся) и время работы под ответом (по умолчанию
показывается).

Настройки у каждого пользователя свои и лежат в таблице user_settings.

Экран настроек — одно сообщение, которое переписывается на месте при
каждом нажатии (messages.edit). Если кнопки у клиента текстовые или
сообщение старше суток (VK не даёт его править), приходит новое.

  • прозрачный фон — картинку с ним нельзя слать как фото: VK пережмёт
    её в JPEG, и прозрачность станет чёрным прямоугольником. Такие
    картинки уходят документом (см. handlers/translate.py);
  • совпавшие цвета текста и фона — рисовать нечитаемый прямоугольник
    бессмысленно, поэтому рендер откатывается на чёрное по белому и
    честно сообщает об этом под картинкой.
"""

import logging

from core import DEFAULT_PUNCT, FONT_SIZES, FONTS, PUNCT_MODES, ImageOptions
from core.todo_image import BG_PALETTE, color_label

from .. import keyboards, texts
from ..chat import Chat
from ..storage import Storage

log = logging.getLogger(__name__)

_VALID_COLORS = {key for key, _, _ in BG_PALETTE}


async def load_options(storage: Storage, user_id: int) -> ImageOptions:
    """Настройки пользователя -> ImageOptions. Незаполненные поля берут
    значения по умолчанию, мусор из БД игнорируется."""
    saved = await storage.get_settings(user_id)
    opts = ImageOptions()
    if saved.get("fg") in _VALID_COLORS:
        opts.fg = saved["fg"]
    if saved.get("bg") in _VALID_COLORS:
        opts.bg = saved["bg"]
    if saved.get("size") in FONT_SIZES:
        opts.size = saved["size"]
    if saved.get("font") in FONTS:
        opts.font = saved["font"]
    return opts


async def load_fix_letters(storage: Storage, user_id: int) -> bool:
    """Исправлять ли текст без калмыцких букв. По умолчанию — нет."""
    return (await storage.get_settings(user_id)).get("fix_letters") == "on"


async def load_punctuation(storage: Storage, user_id: int) -> str:
    """Знаки препинания тодо бичиг: off | frame | all. По умолчанию — off."""
    saved = (await storage.get_settings(user_id)).get("punctuation")
    return saved if saved in PUNCT_MODES else DEFAULT_PUNCT


async def load_show_time(storage: Storage, user_id: int) -> bool:
    """Показывать ли время работы под ответом. По умолчанию — да."""
    return (await storage.get_settings(user_id)).get("show_time") != "off"


def _describe(opts: ImageOptions, fix_letters: bool, punct: str, show_time: bool) -> str:
    return texts.SETTINGS.format(
        fg=color_label(opts.fg),
        bg=color_label(opts.bg),
        size=texts.SIZE_LABELS.get(opts.size, opts.size),
        font=texts.FONT_LABELS.get(opts.font, opts.font),
        fix_letters=texts.FIX_LETTERS_ON if fix_letters else texts.FIX_LETTERS_OFF,
        punctuation=texts.PUNCT_LABELS.get(punct, punct),
        show_time=texts.SHOW_TIME_ON if show_time else texts.SHOW_TIME_OFF,
    )


async def _load_all(storage: Storage, user_id: int):
    return (
        await load_options(storage, user_id),
        await load_fix_letters(storage, user_id),
        await load_punctuation(storage, user_id),
        await load_show_time(storage, user_id),
    )


async def cmd_settings(chat: Chat, storage: Storage) -> None:
    opts, fix, punct, show_time = await _load_all(storage, chat.user_id)
    markup = keyboards.settings_menu(chat.client, opts, fix, punct, show_time)
    text = _describe(opts, fix, punct, show_time)
    await chat.send(text + ("" if markup else texts.SETTINGS_NO_BUTTONS), markup)


async def _refresh(chat: Chat, opts, fix, punct, show_time, markup=None) -> None:
    """Перерисовать экран настроек. Без markup — главное меню."""
    if markup is None:
        markup = keyboards.settings_menu(chat.client, opts, fix, punct, show_time)
    await chat.edit_or_send(_describe(opts, fix, punct, show_time), markup)


async def on_settings(chat: Chat, payload: dict, storage: Storage) -> None:
    action = payload.get("do", "")
    field = str(payload.get("f", ""))
    user_id = chat.user_id
    opts, fix, punct, show_time = await _load_all(storage, user_id)

    if action == "pick":
        if field == "size":
            markup = keyboards.size_picker(chat.client, opts.size)
        elif field == "font":
            markup = keyboards.font_picker(chat.client, opts.font)
        elif field == "punctuation":
            markup = keyboards.punct_picker(chat.client, punct)
        elif field in ("fg", "bg"):
            page = payload.get("p", 0)
            markup = keyboards.color_picker(
                chat.client, field, getattr(opts, field), page if isinstance(page, int) else 0
            )
        else:
            await chat.answer()
            return
        await chat.answer()
        await _refresh(chat, opts, fix, punct, show_time, markup)
        return

    if action == "toggle" and field == "fix_letters":
        fix = not fix
        await storage.set_setting(user_id, "fix_letters", "on" if fix else "off")
        await chat.notify(texts.SETTINGS_FIX_ON if fix else texts.SETTINGS_FIX_OFF, optional=True)
        await _refresh(chat, opts, fix, punct, show_time)
        return

    if action == "toggle" and field == "show_time":
        show_time = not show_time
        await storage.set_setting(user_id, "show_time", "on" if show_time else "off")
        await chat.notify(texts.SETTINGS_TIME_ON if show_time else texts.SETTINGS_TIME_OFF, optional=True)
        await _refresh(chat, opts, fix, punct, show_time)
        return

    if action == "set":
        value = str(payload.get("v", ""))
        ok = ((field in ("fg", "bg") and value in _VALID_COLORS)
              or (field == "size" and value in FONT_SIZES)
              or (field == "font" and value in FONTS)
              or (field == "punctuation" and value in PUNCT_MODES))
        if not ok:
            await chat.answer()
            return
        await storage.set_setting(user_id, field, value)
        opts = await load_options(storage, user_id)
        punct = await load_punctuation(storage, user_id)
        # предупреждаем сразу, а не когда человек получит чёрно-белую картинку
        if field in ("fg", "bg") and opts.fg == opts.bg:
            await chat.notify(texts.SETTINGS_SAME_COLOR, alert=True)
        else:
            await chat.notify(texts.SETTINGS_SAVED, optional=True)
        await _refresh(chat, opts, fix, punct, show_time)
        return

    if action == "back":
        await chat.answer()
        await _refresh(chat, opts, fix, punct, show_time)
        return

    if action == "reset":
        await storage.reset_settings(user_id)
        await chat.notify(texts.SETTINGS_RESET, optional=True)
        await _refresh(chat, ImageOptions(), False, DEFAULT_PUNCT, True)
        return

    await chat.answer()
