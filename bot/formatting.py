# -*- coding: utf-8 -*-
"""Сборка текста ответа: результат + «вход распознан как…» + время работы.

Всё — плоский текст: сообщения сообществ ВКонтакте разметку не
поддерживают, теги показались бы как есть.
"""

from core import Result

from . import texts

TARGET_ICONS = {"translit": "🔤", "todo": "ᡐ", "image": "🖼", "fix": "✏️"}

MESSAGE_LIMIT = 4096


def fmt_ms(ms: float) -> str:
    """412.7 -> «413 мс», 1234.5 -> «1,23 с» — читается легче, чем голые ms."""
    if ms < 1000:
        return f"{ms:.0f} мс"
    return f"{ms / 1000:.2f} с".replace(".", ",")


def timing_line(res) -> str:
    """Только общее время. Разбивка по шагам — внутренняя кухня: она есть
    в БД для анализа, но человеку в чате не нужна."""
    return f"⏱ {fmt_ms(res.elapsed_ms)}"


def _header(res: Result) -> str:
    icon = TARGET_ICONS.get(res.target, "•")
    title = texts.MODE_TITLES.get(res.target, res.target).split(" ", 1)[-1]
    return f"{icon} {title}"


def render_result(res: Result, show_time: bool = True) -> str:
    """Текст ответа для режимов «транслитерация», «тодо бичиг», «исправление»."""
    if res.target == "fix":
        return _render_fix(res, show_time)

    blocks = [_header(res)]
    if res.target == "translit" and res.cyrillic:
        # тодо бичиг или латиница -> кириллица; транслитерацию латиницы
        # повторять незачем
        if res.source_script != "translit":
            blocks.append(f"{texts.LABEL_TRANSLIT}\n{res.translit or ''}")
        blocks.append(f"{texts.LABEL_CYRILLIC}\n{res.cyrillic}")
    elif res.target == "translit":
        blocks.append(res.translit or "")
    else:
        if res.translit:
            blocks.append(f"{texts.LABEL_TRANSLIT}\n{res.translit}")
        blocks.append(f"{texts.LABEL_TODO}\n{res.todo or ''}")

    note = letters_note(res)
    if note:
        blocks.append(note)
    if show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def _render_fix(res: Result, show_time: bool = True) -> str:
    """Режим исправления: исправленный текст целиком (его удобно
    скопировать) и что именно поменялось."""
    blocks = [_header(res)]
    if res.letter_fixes:
        blocks.append(f"{texts.FIX_TITLE}\n{res.fixed_text or ''}")
        blocks.append(letters_note(res))
    else:
        blocks.append(texts.FIX_NOTHING)
    if show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


# сколько исправленных слов показывать, остальные — «и ещё N»
LETTER_FIXES_SHOWN = 12


def letters_note(res: Result) -> str:
    """Что сделано с калмыцкими буквами: список исправлений, если
    исправление включено, или подсказка про настройки, если выключено, а
    текст похож на набранный без ә ө ү һ җ ң."""
    if res.letter_fixes:
        shown = res.letter_fixes[:LETTER_FIXES_SHOWN]
        pairs = ", ".join(f"{a} → {b}" for a, b in shown)
        text = texts.LETTERS_FIXED.format(pairs=pairs)
        if len(res.letter_fixes) > len(shown):
            text += texts.LETTERS_FIXED_MORE.format(n=len(res.letter_fixes) - len(shown))
        return text
    if res.letters_flag:
        return texts.LETTERS_HINT
    return ""


def image_caption(
    res: Result, first: int, last: int, total: int,
    show_time: bool = True, final: bool = True,
) -> str:
    """Текст к сообщению с картинками first..last из total.

    В VK к одному сообщению прикрепляется до 10 картинок, поэтому листы
    идут пачками; служебное (предупреждения, исправленные буквы) — под
    первой пачкой, время — под последней."""
    header = _header(res)
    if total > last - first + 1:
        header += texts.IMAGE_PAGES.format(first=first, last=last, total=total)
    blocks = [header]

    if first == 1:
        if not res.shaping_ok:
            from core.todo_image import SHAPING_WARNING

            blocks.append(SHAPING_WARNING)
        if res.color_fallback:
            blocks.append(texts.COLOR_FALLBACK_NOTE)
        if res.pages_dropped:
            blocks.append(texts.PAGES_DROPPED.format(n=res.pages_dropped))
        note = letters_note(res)
        if note:
            blocks.append(note)

    if final and show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def columns_word(n: int) -> str:
    """3 -> «3 столбца»: 1 столбец, 2–4 столбца, 5–20 столбцов, 21 столбец..."""
    if n % 10 == 1 and n % 100 != 11:
        word = "столбец"
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = "столбца"
    else:
        word = "столбцов"
    return f"{n} {word}"


def render_ocr(res, show_time: bool = True) -> list:
    """Ответ на фото: кириллица, транслитерация и тодо бичиг, по строке на столбец.

    Полная страница — это около 2000 символов на все записи, почти всегда
    одно сообщение. Если не влезает в лимит VK, режем по столбцам: каждая
    часть начинается со своего заголовка. Возвращает список текстов."""
    blocks = [texts.OCR_TITLE.format(n=len(res.columns))]
    if res.cyrillic_columns:
        blocks += _line_blocks(texts.LABEL_CYRILLIC, res.cyrillic_columns)
    blocks += _line_blocks(texts.LABEL_TRANSLIT, res.translit_columns)
    blocks += _line_blocks(texts.LABEL_TODO, res.columns)
    if res.unreadable:
        blocks.append(
            texts.OCR_UNREADABLE.format(cols=columns_word(res.unreadable))
            + (texts.OCR_UNREADABLE_OVERLAY if res.overlay else "")
            + texts.OCR_UNREADABLE_HINT
        )
    if res.low_confidence:
        blocks.append(texts.OCR_LOW_CONFIDENCE)
    if show_time:
        blocks.append(f"⏱ {fmt_ms(res.elapsed_ms)}")
    return _pack(blocks, MESSAGE_LIMIT)


def _line_blocks(title: str, lines: list, limit: int = 3500) -> list:
    """Строки -> блоки «заголовок + строки» не длиннее limit."""
    chunks, cur = [], []
    for line in lines:
        if cur and len("\n".join(cur + [line])) > limit:
            chunks.append(cur)
            cur = []
        cur.append(line)
    chunks.append(cur)
    return [
        (title + "\n" if k == 0 else "") + "\n".join(chunk)
        for k, chunk in enumerate(chunks)
    ]


def _pack(blocks: list, limit: int) -> list:
    """Блоки -> сообщения: подряд, пока влезает в limit."""
    messages = []
    for block in blocks:
        if messages and len(messages[-1]) + 2 + len(block) <= limit:
            messages[-1] += "\n\n" + block
        else:
            messages.append(block)
    return messages
