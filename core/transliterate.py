# -*- coding: utf-8 -*-
"""
transliterate.py — калмыцкая кириллица в транслитерацию проекта.
Работает и на отдельном слове, и на целом предложении.

Как это устроено:
  1. Текст режется на слова (дефис внутри слова не разделяет —
     «Моңһл-Күрә» один токен). Пробелы, пунктуация, цифры не трогаются.
  2. Слово целиком ищется в точном словаре проверенных пар: там, где он
     есть, он надёжнее модели.
  3. Составное слово через дефис разбирается по частям — но только если
     каждая часть сама есть в словаре. Подробности ниже, у _translit_token.
  4. Остальное предсказывает модель — посимвольный seq2seq на NumPy,
     см. core/model_numpy.py.
  5. Регистр восстанавливается отдельно: модель обучена на строчных.
     Бот этим не пользуется — pipeline приводит кириллицу к строчным ещё
     до вызова, потому что заглавная первая буква давала другую запись
     («Җирһл» -> J̌irγal с отдельным диакритическим знаком). Здесь
     восстановление регистра оставлено для прямых вызовов из кода.
"""

import os
import re
import unicodedata
from pathlib import Path

from .model_numpy import load

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = _HERE.parent / "model" / "translit_model.npz"
MODEL_PATH = os.environ.get("MODEL_PATH") or str(_DEFAULT_MODEL)


def get_bundle():
    """Загруженная модель (грузится при первом обращении)."""
    return load(os.environ.get("MODEL_PATH") or MODEL_PATH)


def warmup():
    """Прогреть заранее, чтобы первый пользователь не ждал загрузку весов."""
    transliterate_word("хальмг")


def model_info():
    m = get_bundle()
    return {
        "backend": "numpy",
        "dictionary_size": len(m.dictionary),
        "src_vocab": len(m.src_vocab),
        "tgt_vocab": len(m.tgt_vocab),
        "max_len": m.max_len,
        "model_path": m.path,
    }


def transliterate_word(word):
    return get_bundle().translate_word(word)


# =============================================================================
# Предложение: токенизация + словарь/модель + регистр
# =============================================================================

_WORD_RE = re.compile(r"[а-яёәөүһҗңa-z]+(?:-[а-яёәөүһҗңa-z]+)*", re.IGNORECASE)

# Чем соединять части составного слова, разобранного по дефису.
# Обычный пробел: көвүн-күүкн — это два полноценных слова, и в письме они
# разделяются широким пробелом, а не узким неразрывным (узкий,  , у нас
# означает границу суффикса и получается из дефиса — см. translit_todo.py).
COMPOUND_JOINER = " "

# Дефис в калмыцкой орфографии означает две разные вещи, и по нему одному
# их не различить:
#
#   составное слово — две самостоятельные основы: көвүн-күүкн, эк-эцк,
#     келн-мелн. Их надо разбирать по отдельности: модель на целом токене
#     выдаёт мусор (көвүн-күүкн -> köböüngöükken);
#
#   суффикс через дефис — һазр-ән, кел-ән. Тут делить нельзя: «ән»
#     отдельным словом не существует, а целый токен модель обрабатывает
#     верно (γazar-bēn против неверного γazar-ani при делении).
#
# Раньше условием деления было «обе части есть в словаре». Это ломалось на
# парах вроде келн-мелн: «мелн» — эхо-слово, в словаре его нет и не будет,
# и токен целиком уходил в модель. Поэтому теперь часть считается
# самостоятельным словом, если она либо есть в словаре, либо просто
# достаточно длинная: суффиксы, пишущиеся через дефис, короткие.
#
# Порог легко подвинуть. Если найдётся трёхбуквенный суффикс с дефисом,
# добавьте его в HYPHEN_SUFFIXES — список проверяется раньше длины.
HYPHEN_SUFFIX_MAX_LEN = 2
HYPHEN_SUFFIXES: set = set()


def _is_standalone_word(part, dictionary):
    """Похожа ли часть до/после дефиса на самостоятельное слово."""
    if part in HYPHEN_SUFFIXES:
        return False
    if part in dictionary:
        return True
    return len(part) > HYPHEN_SUFFIX_MAX_LEN


def _restore_case(original_token, result):
    if original_token.isupper() and len(original_token) > 1:
        return result.upper()
    if original_token[:1].isupper():
        return result[:1].upper() + result[1:]
    return result


def _translit_token(token):
    """Одно «слово» из текста -> транслитерация. Возвращает (результат,
    откуда взялось): dict — точный словарь, split — составное слово,
    разобранное по дефису, model — предсказание модели."""
    m = get_bundle()
    key = unicodedata.normalize("NFC", token.lower())

    # 1. весь токен целиком есть в словаре — самый надёжный ответ,
    #    там уже лежат и зер-зев, и моңһл-күрә
    result = m.dictionary.get(key)
    if result is not None:
        return _restore_case(token, result), "dict"

    # 2. составное слово через дефис: көвүн-күүкн, эк-эцк, ах-дү.
    #    Модель на таком токене целиком выдаёт мусор (көвүн-күүкн ->
    #    köböüngöükken), а по частям всё чисто. Но делим ТОЛЬКО когда
    #    каждая часть сама по себе есть в словаре как самостоятельное
    #    слово — иначе так же разобрался бы и суффикс: в һазр-ән «ән»
    #    отдельным словом не существует, и целый токен модель обрабатывает
    #    правильно (γazar-bēn), а по частям вышло бы неверное γazar-ani.
    if "-" in key:
        parts = [p for p in key.split("-") if p]
        if len(parts) > 1 and all(
            _is_standalone_word(p, m.dictionary) for p in parts
        ):
            # каждую часть проводим через ту же развилку «словарь или
            # модель»: в келн-мелн первая часть словарная, вторая нет
            pieces = [
                m.dictionary.get(p) or m.translate_word(p) for p in parts
            ]
            return _restore_case(token, COMPOUND_JOINER.join(pieces)), "split"

    # 3. всё остальное — модель, на токене целиком (суффиксы через дефис
    #    попадают сюда и обрабатываются как надо)
    return _restore_case(token, m.translate_word(key)), "model"


def transliterate(text):
    """Кириллический текст (слово или предложение) -> латиница проекта."""
    return transliterate_with_stats(text)[0]


def transliterate_with_stats(text):
    """То же самое плюс статистика по источнику каждого слова:
    (результат, {"dict": n, "split": k, "model": m})."""
    out_parts = []
    stats = {"dict": 0, "split": 0, "model": 0}
    pos = 0
    for match in _WORD_RE.finditer(text):
        out_parts.append(text[pos:match.start()])
        piece, source = _translit_token(match.group())
        stats[source] += 1
        out_parts.append(piece)
        pos = match.end()
    out_parts.append(text[pos:])
    return "".join(out_parts), stats


if __name__ == "__main__":
    print("Модель:", model_info())
    for w in ["альт", "тертцхн", "хальмг", "медвч"]:
        print(f"  {w} -> {transliterate_word(w)}")
    sentence = "Хальмг улс һазр-ән наран үдэ хәләҗи бәрдг."
    print("\n ", sentence)
    print(" ", transliterate(sentence))
