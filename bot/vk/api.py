# -*- coding: utf-8 -*-
"""
Тонкий клиент VK API: вызов методов, загрузка фото и документов в
сообщения, скачивание вложений.

Фреймворк (vkbottle и т.п.) сознательно не используется: боту нужно с
десяток методов, а своя обёртка в пару сотен строк прозрачна, не тянет
зависимостей и не ломается от чужих мажорных версий.

Ограничения VK, вокруг которых всё устроено:

  • ключ сообщества — не больше 20 запросов в секунду. Ответ на одно
    сообщение — это 2–4 вызова (setActivity, upload, save, send), а лист
    картинок — до 20 загрузок, поэтому вызовы проходят через ограничитель
    частоты, а ошибка 6 «слишком много запросов» повторяется;
  • messages.send требует random_id: VK по нему отбрасывает дубли, так
    что повтор после обрыва сети не пришлёт ответ дважды.
"""

import asyncio
import json
import logging
import random
import time
from collections import deque
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"

# коды ошибок VK, после которых есть смысл повторить запрос
_RETRY_CODES = {
    1,   # Unknown error
    6,   # Too many requests per second
    10,  # Internal server error
}


class VkApiError(Exception):
    def __init__(self, method: str, error: dict):
        self.method = method
        self.code = int(error.get("error_code", 0))
        self.msg = error.get("error_msg", "")
        super().__init__(f"{method}: [{self.code}] {self.msg}")


class UploadError(Exception):
    """Сервер загрузки не принял файл: размеры, формат, вес."""


class _RateLimiter:
    """Не больше `rate` вызовов за любую секунду."""

    def __init__(self, rate: int):
        self.rate = rate
        self._stamps = deque()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 1.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.rate:
                    break
                await asyncio.sleep(1.0 - (now - self._stamps[0]) + 0.01)
            self._stamps.append(time.monotonic())


def _flatten(params: dict) -> dict:
    """Параметры запроса в вид, который понимает VK: списки через
    запятую, словари — JSON, булевы — 1/0, None выкидываем."""
    out = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        out[key] = str(value)
    return out


def random_id() -> int:
    return random.getrandbits(31)


class VkApi:
    def __init__(
        self, token: str, version: str = API_VERSION, rate: int = 19, api_url: str = API_URL
    ):
        self.token = token
        self.api_url = api_url
        self.version = version
        self._limiter = _RateLimiter(rate)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    # ---------------------------------------------------------------- методы

    async def call(self, method: str, **params):
        data = _flatten(params)
        data["access_token"] = self.token
        data["v"] = self.version
        delay = 0.5
        for attempt in range(4):
            await self._limiter.wait()
            try:
                async with self.session.post(self.api_url + method, data=data) as resp:
                    body = await resp.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 3:
                    raise
                log.warning("%s: сеть (%s), повтор через %.1f с", method, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if "error" in body:
                err = VkApiError(method, body["error"])
                if err.code in _RETRY_CODES and attempt < 3:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise err
            return body.get("response")
        raise RuntimeError("unreachable")

    # --------------------------------------------------------- вложения

    async def _post_file(self, url: str, field: str, data: bytes, filename: str) -> dict:
        form = aiohttp.FormData()
        form.add_field(field, data, filename=filename)
        async with self.session.post(url, data=form) as resp:
            return await resp.json(content_type=None)

    async def upload_photo(self, peer_id: int, data: bytes, filename: str) -> str:
        """Картинка -> строка вложения «photo<owner>_<id>_<key>».

        VK принимает в сообщения JPG/PNG/GIF, сумма сторон не больше 14000,
        соотношение сторон не больше 1:20. Не подошло — UploadError, и
        вызывающий шлёт файл документом."""
        server = await self.call("photos.getMessagesUploadServer", peer_id=peer_id)
        up = await self._post_file(server["upload_url"], "photo", data, filename)
        if not up.get("photo") or up.get("photo") == "[]":
            raise UploadError(f"фото не принято: {up.get('error') or up}")
        saved = await self.call(
            "photos.saveMessagesPhoto",
            server=up["server"], photo=up["photo"], hash=up["hash"],
        )
        photo = saved[0]
        ref = f"photo{photo['owner_id']}_{photo['id']}"
        if photo.get("access_key"):
            ref += f"_{photo['access_key']}"
        return ref

    async def upload_doc(
        self, peer_id: int, data: bytes, filename: str, title: Optional[str] = None
    ) -> str:
        """Файл -> строка вложения «doc<owner>_<id>». Так уходят картинки
        с прозрачным фоном (фото VK пережимает в JPEG) и выгрузки CSV."""
        server = await self.call(
            "docs.getMessagesUploadServer", type="doc", peer_id=peer_id
        )
        up = await self._post_file(server["upload_url"], "file", data, filename)
        if not up.get("file"):
            raise UploadError(f"документ не принят: {up.get('error') or up}")
        saved = await self.call("docs.save", file=up["file"], title=title or filename)
        # 5.199 отдаёт {"type": "doc", "doc": {...}}, старые версии — список
        doc = saved.get("doc") if isinstance(saved, dict) else saved[0]
        ref = f"doc{doc['owner_id']}_{doc['id']}"
        if doc.get("access_key"):
            ref += f"_{doc['access_key']}"
        return ref

    async def download(self, url: str, limit: int) -> bytes:
        """Скачать вложение. Больше limit байт — ValueError: картинку на
        сотню мегабайт держать в памяти незачем."""
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            if resp.content_length and resp.content_length > limit:
                raise ValueError("слишком большой файл")
            chunks, total = [], 0
            async for chunk in resp.content.iter_chunked(1 << 16):
                total += len(chunk)
                if total > limit:
                    raise ValueError("слишком большой файл")
                chunks.append(chunk)
        return b"".join(chunks)
