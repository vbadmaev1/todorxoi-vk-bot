# -*- coding: utf-8 -*-
"""
Chat — куда и как отвечать. Хэндлеры не зовут VK API напрямую, а говорят
chat.send(...), chat.notify(...), chat.edit(...): так в одном месте
решается то, чем VK отличается от Telegram.

  • Нажатие callback-кнопки (ButtonPress) обязательно нужно «закрыть»
    messages.sendMessageEventAnswer — иначе у человека бесконечно крутится
    индикатор на кнопке. Короткий ответ показывается всплывающей плашкой
    (snackbar, до 90 символов); длинный уходит обычным сообщением.
  • Старые клиенты не умеют callback-кнопки — им те же кнопки шлются
    текстовыми, и нажатие приходит обычным сообщением с payload. Тогда
    «отредактировать сообщение с кнопкой» нельзя (не знаем, какое) — бот
    просто присылает новое.
  • Текст сообщения — до 4096 символов, вложений — до 10.
"""

import logging
from typing import Iterable, Optional, Sequence

from .vk.api import UploadError, VkApi, VkApiError, random_id
from .vk.events import ClientInfo

log = logging.getLogger(__name__)

MESSAGE_LIMIT = 4096
ATTACHMENTS_LIMIT = 10
SNACKBAR_LIMIT = 90


class Chat:
    def __init__(
        self,
        api: VkApi,
        group_id: int,
        peer_id: int,
        user_id: int,
        client: ClientInfo,
        event_id: Optional[str] = None,
        cmid: Optional[int] = None,
    ):
        self.api = api
        self.group_id = group_id
        self.peer_id = peer_id
        self.user_id = user_id
        self.client = client
        # нажатие callback-кнопки: id события и сообщение, под которым кнопка
        self.event_id = event_id
        self.cmid = cmid
        self.answered = event_id is None

    @property
    def is_press(self) -> bool:
        return self.event_id is not None

    # ------------------------------------------------------------ отправка

    async def send(
        self,
        text: str = "",
        keyboard: Optional[str] = None,
        attachments: Sequence[str] = (),
    ) -> None:
        """Одно сообщение. Слишком длинный текст режется по строкам;
        клавиатура и вложения — у последнего куска."""
        chunks = split_text(text) if text else [""]
        for k, chunk in enumerate(chunks):
            last = k == len(chunks) - 1
            await self.api.call(
                "messages.send",
                peer_id=self.peer_id,
                random_id=random_id(),
                message=chunk or None,
                keyboard=keyboard if last else None,
                attachment=list(attachments) if (last and attachments) else None,
                # без карточки-превью под ссылкой на канал в /help
                dont_parse_links=1,
                disable_mentions=1,
            )

    async def typing(self) -> None:
        try:
            await self.api.call(
                "messages.setActivity",
                peer_id=self.peer_id, type="typing", group_id=self.group_id,
            )
        except VkApiError as exc:
            log.debug("setActivity: %s", exc)

    async def edit(self, text: str, keyboard: Optional[str] = None) -> bool:
        """Переписать сообщение, под которым нажали кнопку (экран
        настроек). False — редактировать нечего или VK не дал (сообщению
        больше суток) — тогда вызывающий шлёт новое."""
        if not self.cmid:
            return False
        try:
            await self.api.call(
                "messages.edit",
                peer_id=self.peer_id,
                conversation_message_id=self.cmid,
                message=text,
                keyboard=keyboard,
                dont_parse_links=1,
                disable_mentions=1,
            )
            return True
        except VkApiError as exc:
            log.info("не удалось отредактировать сообщение: %s", exc)
            return False

    async def edit_or_send(self, text: str, keyboard: Optional[str] = None) -> None:
        if not await self.edit(text, keyboard):
            await self.send(text, keyboard)

    # ------------------------------------------------- ответ на нажатие

    async def answer(self, text: Optional[str] = None) -> None:
        """Закрыть нажатие callback-кнопки: снять индикатор и, если есть
        текст, показать плашку. Для обычных сообщений — ничего."""
        if self.answered:
            return
        self.answered = True
        event_data = None
        if text:
            event_data = {"type": "show_snackbar", "text": text[:SNACKBAR_LIMIT]}
        try:
            await self.api.call(
                "messages.sendMessageEventAnswer",
                event_id=self.event_id,
                user_id=self.user_id,
                peer_id=self.peer_id,
                event_data=event_data,
            )
        except VkApiError as exc:
            # событие устарело (ответ дольше ~1 минуты) — не страшно
            log.info("sendMessageEventAnswer: %s", exc)

    async def notify(self, text: str, alert: bool = False, optional: bool = False) -> None:
        """Подтверждение нажатия. На callback-кнопку — плашкой; если текст
        важный (alert) или не влезает в плашку — сообщением. На текстовую
        кнопку плашек нет — сообщением, кроме optional: «Сохранил» в
        настройках видно и по обновлённому экрану, отдельное сообщение на
        каждое нажатие только засоряло бы диалог."""
        if self.is_press and not alert and len(text) <= SNACKBAR_LIMIT:
            await self.answer(text)
            return
        await self.answer()
        if optional and not alert and not self.is_press:
            return
        await self.send(text)

    # --------------------------------------------------------- вложения

    async def upload(self, data: bytes, filename: str, as_doc: bool = False) -> str:
        """Картинка -> вложение. Фото, если можно; если VK фото не принял
        (размеры, пропорции) — документом."""
        if not as_doc:
            try:
                return await self.api.upload_photo(self.peer_id, data, filename)
            except (UploadError, VkApiError) as exc:
                log.warning("фото не загрузилось (%s) — шлю документом", exc)
        return await self.api.upload_doc(self.peer_id, data, filename)


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list:
    """Текст -> куски не длиннее limit, по возможности по пустым строкам,
    потом по строкам, в крайнем случае — посреди строки."""
    if len(text) <= limit:
        return [text]
    out = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        out.append(rest)
    return out


def batched(items: Iterable, size: int = ATTACHMENTS_LIMIT) -> list:
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]
