# -*- coding: utf-8 -*-
"""
Поддельный VK на localhost: методы API, сервер Long Poll, серверы
загрузки фото и документов, раздача «вложений» по ссылке.

Ведёт себя как VK в том, что важно боту, и проверяет то, за что VK
отказал бы: лимиты клавиатур, длину текста, обязательный random_id,
непустое сообщение. Нарушение — ответ с error, как у настоящего API,
и тест это увидит.
"""

import asyncio
import json
from typing import List

from aiohttp import web

PHOTO_OWNER = -100500
GROUP_ID = 100500


class FakeVk:
    def __init__(self):
        self.calls: List[tuple] = []          # (метод, параметры)
        self.uploads: List[tuple] = []        # (photo|doc, имя файла, байты)
        self.files = {}                       # путь -> байты для скачивания
        self.events = asyncio.Queue()
        self.lp_requests = 0
        self.fail_key_once = True             # первый a_check ответит failed: 2
        self.reject_photo = False             # сервер фото «не принял» картинку
        self._ts = 1
        self._ids = 0
        self.runner = None
        self.base = ""

    # ------------------------------------------------------------ запуск

    async def start(self) -> None:
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.router.add_post("/method/{name}", self.method)
        app.router.add_get("/lp", self.longpoll)
        app.router.add_post("/upload/{kind}", self.upload)
        app.router.add_get("/files/{name}", self.file)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        await self.runner.cleanup()

    @property
    def api_url(self) -> str:
        return self.base + "/method/"

    def add_file(self, name: str, data: bytes) -> str:
        self.files[name] = data
        return f"{self.base}/files/{name}"

    def sent(self, method: str = "messages.send") -> list:
        return [params for name, params in self.calls if name == method]

    def reset(self) -> None:
        self.calls.clear()
        self.uploads.clear()

    def _id(self) -> int:
        self._ids += 1
        return self._ids

    # ------------------------------------------------------------- API

    @staticmethod
    def _error(code: int, msg: str):
        return web.json_response({"error": {"error_code": code, "error_msg": msg}})

    async def method(self, request: web.Request):
        name = request.match_info["name"]
        params = dict(await request.post())
        if params.get("access_token") != "TEST" or not params.get("v"):
            return self._error(5, "User authorization failed")
        params.pop("access_token")
        params.pop("v")
        self.calls.append((name, params))

        if name == "groups.getById":
            return web.json_response({"response": {"groups": [
                {"id": GROUP_ID, "name": "Тодо бичиг", "screen_name": "todorxoi"}
            ], "profiles": []}})
        if name == "groups.getLongPollServer":
            return web.json_response({"response": {
                "server": self.base + "/lp", "key": f"key{len(self.calls)}", "ts": str(self._ts),
            }})
        if name == "groups.getLongPollSettings":
            return web.json_response({"response": {
                "is_enabled": False, "api_version": "5.131",
                "events": {"message_new": 1, "message_event": 0, "wall_post_new": 0},
            }})
        if name in ("groups.setLongPollSettings", "groups.setSettings"):
            return self._error(15, "Access denied: no access to call this method")
        if name == "messages.send":
            err = self._check_send(params)
            if err:
                return self._error(100, err)
            return web.json_response({"response": self._id()})
        if name == "messages.edit":
            if not params.get("message") or not params.get("conversation_message_id"):
                return self._error(100, "message is empty or cmid missing")
            err = self._check_keyboard(params.get("keyboard"))
            if err:
                return self._error(911, err)
            return web.json_response({"response": 1})
        if name in ("messages.sendMessageEventAnswer", "messages.setActivity"):
            if name == "messages.sendMessageEventAnswer" and params.get("event_data"):
                data = json.loads(params["event_data"])
                if len(data.get("text", "")) > 90:
                    return self._error(100, "snackbar text too long")
            return web.json_response({"response": 1})
        if name == "photos.getMessagesUploadServer":
            return web.json_response({"response": {"upload_url": self.base + "/upload/photo"}})
        if name == "photos.saveMessagesPhoto":
            return web.json_response({"response": [
                {"id": self._id(), "owner_id": PHOTO_OWNER, "access_key": "abc"}
            ]})
        if name == "docs.getMessagesUploadServer":
            return web.json_response({"response": {"upload_url": self.base + "/upload/doc"}})
        if name == "docs.save":
            return web.json_response({"response": {"type": "doc", "doc": {
                "id": self._id(), "owner_id": PHOTO_OWNER, "title": params.get("title"),
            }}})
        return self._error(3, f"Unknown method passed: {name}")

    def _check_send(self, params: dict) -> str:
        if not params.get("random_id"):
            return "random_id is required"
        if not params.get("peer_id"):
            return "peer_id is required"
        if not params.get("message") and not params.get("attachment"):
            return "message is empty"
        if len(params.get("message", "")) > 4096:
            return "message is too long"
        if len((params.get("attachment") or "").split(",")) > 10:
            return "too many attachments"
        return self._check_keyboard(params.get("keyboard"))

    @staticmethod
    def _check_keyboard(raw) -> str:
        if not raw:
            return ""
        kb = json.loads(raw)
        rows = kb["buttons"]
        total = sum(len(r) for r in rows)
        if kb.get("inline"):
            if len(rows) > 6 or total > 10:
                return f"inline keyboard too big: {len(rows)} rows / {total} buttons"
        elif len(rows) > 10 or total > 40:
            return "keyboard too big"
        for row in rows:
            if len(row) > 5:
                return "too many buttons in a row"
            for b in row:
                action = b["action"]
                if len(action.get("label", "")) > 40:
                    return f"label too long: {action['label']}"
                if len(action.get("payload", "").encode()) > 255:
                    return "payload too long"
                json.loads(action["payload"])
        return ""

    # -------------------------------------------------------- Long Poll

    async def longpoll(self, request: web.Request):
        self.lp_requests += 1
        q = request.query
        if q.get("act") != "a_check" or not q.get("key"):
            return web.json_response({"failed": 3})
        if self.fail_key_once:
            self.fail_key_once = False
            return web.json_response({"failed": 2})
        updates = []
        try:
            first = await asyncio.wait_for(self.events.get(), timeout=0.5)
            updates.append(first)
            while not self.events.empty():
                updates.append(self.events.get_nowait())
        except asyncio.TimeoutError:
            pass
        self._ts += 1
        return web.json_response({"ts": str(self._ts), "updates": updates})

    # ---------------------------------------------------------- загрузка

    async def upload(self, request: web.Request):
        kind = request.match_info["kind"]
        reader = await request.multipart()
        part = await reader.next()
        data = await part.read()
        self.uploads.append((kind, part.filename, data))
        if kind == "photo":
            if part.name != "photo":
                return web.json_response({"error": "field must be photo"})
            if self.reject_photo:
                return web.json_response({"server": 1, "photo": "[]", "hash": "x"})
            return web.json_response({"server": 1, "photo": '[{"p":1}]', "hash": "h"})
        if part.name != "file":
            return web.json_response({"error": "field must be file"})
        return web.json_response({"file": "f" * 10})

    async def file(self, request: web.Request):
        data = self.files.get(request.match_info["name"])
        if data is None:
            raise web.HTTPNotFound()
        return web.Response(body=data, content_type="image/png")
