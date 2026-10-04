# -*- coding: utf-8 -*-
"""
Прогон бота «на сухую», без ВКонтакте: поддельный VK поднимается на
localhost (tests/fake_vk.py), бот ходит к нему настоящим HTTP-клиентом,
а события — в том виде, в каком их присылает Long Poll.

Проверяем сценарии целиком: что бот ответил, с какими кнопками и
вложениями, что записал в базу. Поддельный VK отклоняет то же, что
отклонил бы настоящий: слишком большие клавиатуры, длинные подписи
кнопок, сообщения без random_id.

Запуск (ключ не нужен, сеть тоже):

    python -m tests.smoke
"""

import asyncio
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.mkdtemp(prefix="todorxoi-vk-")
os.environ.update({
    "VK_TOKEN": "TEST",
    "ADMIN_IDS": "5",
    "WARMUP": "false",
    "DATA_DIR": _tmp,
    "NEWS_URL": "https://t.me/todorxoi_uzuq",
})

from bot.config import Config  # noqa: E402
from bot.handlers import Dispatcher, parse_command  # noqa: E402
from bot.storage import Storage  # noqa: E402
from bot.vk import longpoll  # noqa: E402
from bot.vk.api import VkApi  # noqa: E402
from tests.fake_vk import GROUP_ID, FakeVk  # noqa: E402

USER = 5
OTHER = 6
CHAT_PEER = 2_000_000_001
FULL_CLIENT = {
    "button_actions": ["text", "vkpay", "open_app", "location", "open_link",
                       "callback", "intent_subscribe", "intent_unsubscribe"],
    "keyboard": True, "inline_keyboard": True, "carousel": True, "lang_id": 0,
}
OLD_CLIENT = {"button_actions": ["text"], "keyboard": True, "inline_keyboard": True}

_cmid = [100]
failures = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def message_new(text="", peer=USER, sender=USER, payload=None, attachments=None,
                client=FULL_CLIENT, reply=None):
    _cmid[0] += 1
    msg = {
        "date": 1759500000, "from_id": sender, "id": 0, "out": 0, "version": 1,
        "attachments": attachments or [], "conversation_message_id": _cmid[0],
        "fwd_messages": [], "important": False, "is_hidden": False,
        "peer_id": peer, "random_id": 0, "text": text,
    }
    if payload is not None:
        msg["payload"] = json.dumps(payload)
    if reply is not None:
        msg["reply_message"] = reply
    return {"type": "message_new", "object": {"message": msg, "client_info": client},
            "group_id": GROUP_ID, "event_id": f"e{_cmid[0]}", "v": "5.199"}


def message_event(payload, peer=USER, user=USER, cmid=777):
    return {"type": "message_event", "group_id": GROUP_ID, "v": "5.199", "object": {
        "user_id": user, "peer_id": peer, "event_id": f"ev{_cmid[0]}",
        "payload": payload, "conversation_message_id": cmid,
    }}


def keyboard_of(params):
    raw = params.get("keyboard")
    return json.loads(raw) if raw else None


def buttons(params):
    kb = keyboard_of(params) or {"buttons": []}
    return [b["action"] for row in kb["buttons"] for b in row]


def payloads(params):
    return [json.loads(b["payload"]) for b in buttons(params)]


def find_button(params, **match):
    for p in payloads(params):
        if all(p.get(k) == v for k, v in match.items()):
            return p
    return None


def last_send(vk):
    sends = vk.sent()
    return sends[-1] if sends else {}


def photo_attachment(url, w=800, h=600):
    return {"type": "photo", "photo": {
        "id": 1, "owner_id": USER, "access_key": "k",
        "sizes": [{"type": "m", "url": url + "?m", "width": 130, "height": 98},
                  {"type": "x", "url": url, "width": w, "height": h}],
    }}


async def scenario(vk: FakeVk, d: Dispatcher, storage: Storage) -> None:
    async def feed(update):
        await d.feed(update)

    print("\nразбор команд")
    check(parse_command("/todo хальмг улс") == ("todo", "хальмг улс"), "/todo с текстом")
    check(parse_command("/todo@todorxoi") == ("todo", ""), "/todo@бот")
    check(parse_command("Начать") == ("start", ""), "«Начать» без слэша")
    check(parse_command("/nope") == (None, ""), "незнакомая команда")
    check(parse_command("хальмг улс") is None, "обычный текст — не команда")
    check(parse_command("/fix бяядл\nкюн") == ("fix", "бяядл\nкюн"), "многострочный аргумент")

    print("\n«Начать» и меню")
    vk.reset()
    await feed(message_new("Начать", payload={"command": "start"}))
    s = last_send(vk)
    check("Тодо Бичик бот" in s.get("message", ""), "приветствие")
    kb = keyboard_of(s)
    check(kb and not kb["inline"] and len(kb["buttons"]) == 4, "нижнее меню из 4 рядов")
    colors = {b["action"]["label"]: b.get("color") for row in kb["buttons"] for b in row}
    check(colors.get("🔤 Транслитерация") == "primary", "текущий режим подсвечен")
    check(s.get("dont_parse_links") == "1", "без превью ссылок")
    check(any(m == "messages.setActivity" for m, _ in vk.calls) is False,
          "на /start нет «печатает…»")

    print("\nтекст в режиме по умолчанию (транслитерация)")
    vk.reset()
    await feed(message_new("Хальмг Таңһч"))
    s = last_send(vk)
    check("xalimaq" in s.get("message", "").lower(), "транслитерация в ответе")
    check(any(m == "messages.setActivity" for m, _ in vk.calls), "«печатает…»")
    conv = find_button(s, a="conv", t="todo")
    fb_up = find_button(s, a="fb", v="up")
    check(conv is not None and fb_up is not None, "кнопки «→ Тодо» и 👍/👎")
    check(keyboard_of(s)["inline"], "кнопки под сообщением — inline")
    check(buttons(s)[0]["type"] == "callback", "callback-кнопки у нового клиента")
    rid = conv["r"]
    row = await storage.get_request(rid)
    check(row and row["chat_id"] == USER and row["translit"], "запрос записан в БД")

    print("\nкнопка «→ Тодо бичик»")
    vk.reset()
    await feed(message_event({"a": "conv", "t": "todo", "r": rid}))
    answers = vk.sent("messages.sendMessageEventAnswer")
    check(len(answers) == 1, "нажатие закрыто ровно один раз")
    s = last_send(vk)
    check("Тодо бичик:" in s.get("message", ""), "ответ в тодо бичик")
    check(any("᠀" <= ch <= "᢯" for ch in s.get("message", "")), "в ответе монгольские буквы")

    print("\n👍 и 👎 с исправлением")
    vk.reset()
    await feed(message_event({"a": "fb", "v": "up", "r": rid}))
    ans = vk.sent("messages.sendMessageEventAnswer")
    check(ans and "Спасибо" in json.loads(ans[0]["event_data"])["text"], "плашка «спасибо»")
    check(not vk.sent(), "лишних сообщений нет")

    vk.reset()
    await feed(message_event({"a": "fb", "v": "down", "r": rid}))
    check("правильный ответ" in last_send(vk).get("message", ""), "просит правильный вариант")
    vk.reset()
    await feed(message_new("ᡍᠠᠯᡅᡏᠠᡎ"))
    check("исправление сохранил" in last_send(vk).get("message", ""), "исправление принято")
    csv = (await storage.export_csv(only_corrections=True)).decode("utf-8-sig")
    check("ᡍᠠᠯᡅᡏᠠᡎ" in csv, "исправление в выгрузке")
    vk.reset()
    await feed(message_new("/cancel"))
    check("нечего отменять" in last_send(vk).get("message", ""), "/cancel без ожидания")

    print("\n👎, затем команда — ожидание отменяется")
    await feed(message_event({"a": "fb", "v": "down", "r": rid}))
    vk.reset()
    await feed(message_new("/mode"))
    await feed(message_new("хальмг"))
    check("xalimaq" in last_send(vk).get("message", "").lower(),
          "после команды текст снова переводится")

    print("\nрежим кнопкой меню и с текстом в команде")
    vk.reset()
    await feed(message_new("ᡐ Тодо бичик", payload={"cmd": "todo"}))
    check("Режим: ᡐ Тодо бичик" in last_send(vk).get("message", ""), "режим переключён")
    check(await storage.get_mode(USER) == "todo", "режим сохранён в БД")
    vk.reset()
    await feed(message_new("хальмг улс"))
    check("Тодо бичик:" in last_send(vk).get("message", ""), "текст обработан в новом режиме")

    vk.reset()
    await feed(message_new("/fix Сян бяянт"))
    s = last_send(vk)
    check("бәәнт" in s.get("message", "").lower(), "/fix вернул калмыцкие буквы")
    conv = find_button(s, a="conv", t="translit")
    check(conv is not None, "кнопка «→ Транслитерация» под исправлением")
    vk.reset()
    await feed(message_event(conv))
    check("bayi" in last_send(vk).get("message", "").lower() or
          "bai" in last_send(vk).get("message", "").lower(),
          "по кнопке переводится исправленный текст")

    print("\nкартинка")
    vk.reset()
    await feed(message_new("/image хальмг улс"))
    s = last_send(vk)
    check(s.get("attachment", "").startswith("photo-100500_"), "картинка ушла фото")
    check(vk.uploads and vk.uploads[0][0] == "photo" and vk.uploads[0][2][:4] == b"\x89PNG",
          "на сервер загрузки ушёл PNG")
    check(find_button(s, a="fb", v="up") is not None, "👍/👎 под картинкой")

    long_text = " ".join(["хальмг улсин келн"] * 600)[:9000]
    vk.reset()
    await feed(message_new("/image " + long_text[:4000]))
    sends = vk.sent()
    n_att = [len(p.get("attachment", "").split(",")) for p in sends if p.get("attachment")]
    check(sends and all(n <= 10 for n in n_att), f"листы пачками до 10 ({n_att})")
    check(keyboard_of(sends[-1]) is not None and all(
        keyboard_of(p) is None for p in sends[:-1]), "кнопки только под последней пачкой")

    # 12 листов: в VK до 10 вложений на сообщение — должно уйти двумя
    from types import SimpleNamespace

    from bot.chat import Chat
    from bot.handlers.translate import send_image
    from bot.vk.events import CALLBACK_CLIENT

    one = vk.uploads[0][2]
    fake = SimpleNamespace(
        target="image", pages=[(io.BytesIO(one), (300, 400))] * 12, transparent=False,
        shaping_ok=True, color_fallback=False, pages_dropped=0, letter_fixes=[],
        letters_flag="", elapsed_ms=5.0,
    )
    vk.reset()
    await send_image(Chat(d.api, GROUP_ID, USER, USER, CALLBACK_CLIENT), fake, '{"buttons":[],"inline":true}')
    sends = vk.sent()
    check([len(p["attachment"].split(",")) for p in sends] == [10, 2], "12 листов — сообщениями 10 + 2")
    check("картинки 1–10 из 12" in sends[0]["message"] and "11–12 из 12" in sends[1]["message"],
          "в подписи — какие листы")
    check("⏱" not in sends[0]["message"] and "⏱" in sends[1]["message"], "время — под последним")

    print("\nнастройки: экран, палитра, прозрачный фон")
    vk.reset()
    await feed(message_new("⚙️ Настройки", payload={"cmd": "settings"}))
    s = last_send(vk)
    check("Настройки" in s.get("message", ""), "экран настроек")
    kb = keyboard_of(s)
    check(kb["inline"] and len(kb["buttons"]) <= 6, "меню настроек влезает в 6 рядов")

    vk.reset()
    await feed(message_event({"a": "set", "do": "pick", "f": "bg"}))
    edits = vk.sent("messages.edit")
    check(edits and edits[0]["conversation_message_id"] == "777", "экран правится на месте")
    more = find_button(edits[0], do="pick", f="bg", p=1)
    check(more is not None, "палитра фона листается")
    vk.reset()
    await feed(message_event(more))
    page2 = vk.sent("messages.edit")[0]
    transparent = find_button(page2, do="set", f="bg", v="transparent")
    check(transparent is not None, "прозрачный фон на второй странице")
    vk.reset()
    await feed(message_event(transparent))
    ans = vk.sent("messages.sendMessageEventAnswer")
    check(ans and json.loads(ans[0]["event_data"])["text"] == "Сохранил", "плашка «Сохранил»")
    check("прозрачный" in vk.sent("messages.edit")[0]["message"], "экран обновлён")

    vk.reset()
    await feed(message_new("/image хальмг"))
    check(last_send(vk).get("attachment", "").startswith("doc"), "прозрачный фон — документом")

    vk.reset()
    await feed(message_event({"a": "set", "do": "set", "f": "fg", "v": "white"}))
    await feed(message_event({"a": "set", "do": "set", "f": "bg", "v": "white"}))
    check(any("совпадают" in p.get("message", "") for p in vk.sent()),
          "совпавшие цвета — предупреждение сообщением")

    vk.reset()
    mode_before = await storage.get_mode(USER)
    await feed(message_event({"a": "set", "do": "reset"}))
    check(await storage.get_mode(USER) == mode_before == "image", "сброс настроек не трогает режим")
    check((await storage.get_settings(USER)).get("bg") is None, "фон сброшен")

    print("\nстарый клиент без callback-кнопок")
    vk.reset()
    await feed(message_new("/settings", client=OLD_CLIENT))
    check(buttons(last_send(vk))[0]["type"] == "text", "кнопки текстовые")
    vk.reset()
    await feed(message_new("🔠 Размер: средний", payload={"a": "set", "do": "pick", "f": "size"},
                           client=OLD_CLIENT))
    check(not vk.sent("messages.edit") and find_button(last_send(vk), f="size", v="large"),
          "нажатие текстовой кнопки — новое сообщение с выбором")
    vk.reset()
    await feed(message_new("крупный", payload={"a": "set", "do": "set", "f": "size", "v": "large"},
                           client=OLD_CLIENT))
    check(len(vk.sent()) == 1 and "крупный" in last_send(vk)["message"],
          "без лишнего «Сохранил» — только обновлённый экран")
    vk.reset()
    await feed(message_new("👍", payload={"a": "fb", "v": "up", "r": rid}, client=OLD_CLIENT))
    check("Спасибо" in last_send(vk).get("message", ""), "спасибо за 👍 сообщением")

    print("\nфото → текст")
    from core.todo_image import render_todo_bytes
    from core.translit_todo import translit_to_todo

    buf, _, _ = render_todo_bytes(translit_to_todo("xalimaq tangγači"))
    url = vk.add_file("page.png", buf.getvalue())
    vk.reset()
    await feed(message_new("", attachments=[photo_attachment(url)]))
    s = last_send(vk)
    if "не работает" in s.get("message", ""):
        print("  --   распознавание недоступно в этом окружении (нет onnxruntime/scipy)")
    else:
        check("Текст с фото" in s.get("message", ""), "ответ на фото")
        check(s.get("attachment", "").startswith("photo"), "картинка с рамками в том же сообщении")
        check(find_button(s, a="fb", v="up") is not None, "👍/👎 под ответом на фото")
        check("xalimaq" in s.get("message", "").lower(), "текст распознан")

    vk.reset()
    await feed(message_new("/ocr", reply={"attachments": [photo_attachment(url)], "text": ""}))
    check("Текст с фото" in last_send(vk).get("message", "")
          or "не работает" in last_send(vk).get("message", ""),
          "/ocr ответом на сообщение с фото")
    vk.reset()
    await feed(message_new("", attachments=[{"type": "doc", "doc": {
        "id": 1, "owner_id": USER, "ext": "pdf", "url": url, "size": 10, "title": "a.pdf"}}]))
    check("PDF" in last_send(vk).get("message", ""), "PDF — подсказка")
    vk.reset()
    await feed(message_new("", attachments=[{"type": "sticker", "sticker": {}}]))
    check("понимаю текст и фото" in last_send(vk).get("message", ""), "стикер — подсказка")

    print("\nсервер фото не принял картинку — уходит документом")
    vk.reject_photo = True
    await storage.set_setting(USER, "bg", "white")
    vk.reset()
    await feed(message_new("/image хальмг"))
    check(last_send(vk).get("attachment", "").startswith("doc"), "откат на документ")
    vk.reject_photo = False

    print("\nбеседа")
    vk.reset()
    await feed(message_new("хальмг улс", peer=CHAT_PEER, sender=OTHER))
    check(not vk.sent(), "на обычные реплики в беседе молчит")
    await feed(message_new(f"[club{GROUP_ID}|@todorxoi], хальмг", peer=CHAT_PEER, sender=OTHER))
    check("xalimaq" in last_send(vk).get("message", "").lower(), "отвечает на упоминание")
    vk.reset()
    await feed(message_new("/todo хальмг", peer=CHAT_PEER, sender=OTHER))
    check("Тодо бичик:" in last_send(vk).get("message", ""), "команды в беседе работают")
    check(last_send(vk).get("peer_id") == str(CHAT_PEER), "ответ — в беседу")
    vk.reset()
    await feed(message_new("сообщение", peer=CHAT_PEER, sender=-5))
    check(not vk.sent(), "сообщения других ботов игнорируются")

    print("\nадминка")
    vk.reset()
    await feed(message_new("/stats", sender=OTHER, peer=OTHER))
    check("только администраторам" in last_send(vk).get("message", ""), "чужим /stats нельзя")
    vk.reset()
    await feed(message_new("/stats"))
    check("Статистика" in last_send(vk).get("message", ""), "/stats")
    vk.reset()
    await feed(message_new("/export corrections"))
    s = last_send(vk)
    check(s.get("attachment", "").startswith("doc") and vk.uploads[-1][1] == "corrections.csv",
          "/export — CSV документом")

    print("\nразное")
    vk.reset()
    await feed(message_new("/nope"))
    check("Не знаю такой команды" in last_send(vk).get("message", ""), "незнакомая команда")
    vk.reset()
    await feed(message_new("Помощь", payload={"cmd": "help"}))
    s = last_send(vk)
    check("Что умеет бот" in s.get("message", "") and "t.me/todorxoi_uzuq" in s["message"],
          "помощь со ссылкой на новости")
    vk.reset()
    await feed(message_event({"a": "fb", "v": "up", "r": 999999}))
    check(any("слишком старое" in p.get("message", "") for p in vk.sent()),
          "оценка несуществующего запроса")
    check(len(vk.sent("messages.sendMessageEventAnswer")) == 1, "нажатие всё равно закрыто")
    vk.reset()
    await feed(message_new("12345 !!!"))
    check("⚠️" in last_send(vk).get("message", ""), "нераспознаваемый текст — понятная ошибка")


async def check_longpoll(vk: FakeVk, api: VkApi) -> None:
    print("\nLong Poll")
    await vk.events.put(message_new("хальмг"))
    got = []

    async def consume():
        async for update in longpoll.listen(api, GROUP_ID, wait=1):
            got.append(update)
            return

    await asyncio.wait_for(consume(), timeout=10)
    check(got and got[0]["type"] == "message_new", "событие пришло через Long Poll")
    servers = [p for m, p in vk.calls if m == "groups.getLongPollServer"]
    check(len(servers) == 2, "после failed: 2 взят новый ключ")


async def check_startup(vk: FakeVk, api: VkApi, config: Config) -> None:
    print("\nстарт")
    from bot.main import _group, _setup_group

    group = await _group(api, config)
    check(group["id"] == GROUP_ID, "сообщество определено по ключу")
    await _setup_group(api, GROUP_ID)  # фейк отвечает «нет прав» — бот не падает
    check(True, "без прав на настройку сообщества бот продолжает")


async def main() -> int:
    vk = FakeVk()
    await vk.start()
    config = Config.from_env()
    storage = Storage(config.db_path, config.jsonl_path)
    storage.connect()
    api = VkApi("TEST", api_url=vk.api_url)
    try:
        await check_startup(vk, api, config)
        await check_longpoll(vk, api)
        await scenario(vk, Dispatcher(api, storage, config, GROUP_ID), storage)
    finally:
        storage.close()
        await api.close()
        await vk.stop()

    jsonl = Path(_tmp, "events.jsonl").read_text(encoding="utf-8").splitlines()
    check(len(jsonl) > 10, f"лог событий пишется ({len(jsonl)} строк)")

    print()
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)}")
        for f in failures:
            print("  -", f)
        return 1
    print("Всё в порядке.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
