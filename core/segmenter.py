# -*- coding: utf-8 -*-
"""
segmenter.py — какие куски транслитерации складываются в одно
кириллическое слово.

Одно кириллическое слово в тодо бичиг часто записано несколькими словами:
«орнд» — ᡆᠷᡆᠨ ᡑᡇ (oron du), «келәр» — kele bēr. В транслитерации
распознанного текста все промежутки — пробелы (см. core/ocr.py), и по ним
не видно, где граница слова, а где хвост. Решает эта модель: для каждого
пробела — вероятность «склеить» (внутри одного кириллического слова).

Правилом по списку суффиксов это не решить: одно и то же ügei бывает
отдельным словом (уга) и хвостом (abudaq ügei = авдго), bayinai и bayixu —
примерно поровну. Поэтому модель смотрит на всю строку.

Модель — посимвольная двунаправленная GRU в два слоя (0.9 млн параметров,
model/segmenter_model.npz, float16), обучена в
todorxoi-inference/translit2cyr/segmenter_train.ipynb на корпусе, где
граница слова известна. На val верно решены 99.8 % пробелов.

Формулы GRU — те же, что в core/model_numpy.py (gru_pass).
"""

import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import List

import numpy as np

from .model_numpy import META_KEY, gru_pass

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = _HERE.parent / "model" / "segmenter_model.npz"

UNK_ID = 1                      # itos = [<pad>, <unk>, ...]
_GAPS = re.compile(r"[\s -]+")


def normalize(text: str) -> str:
    """Как строка приходит в модель: нижний регистр, любой промежуток и
    дефис — один пробел. Модель училась ровно на таком виде."""
    text = unicodedata.normalize("NFC", text).lower()
    return _GAPS.sub(" ", text).strip()


class Segmenter:
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(bytes(data[META_KEY]).decode("utf-8"))
        self.path = str(npz_path)
        self.stoi = {ch: i for i, ch in enumerate(meta["itos"])}
        self.hid = meta["hid_dim"]
        self.layers = meta["layers"]
        self.threshold = meta["threshold"]
        w = {k: data[k].astype(np.float32) for k in data.files if k != META_KEY}
        self.emb = w["emb.weight"]
        self.gru = []
        for layer in range(self.layers):
            dirs = []
            for suffix in ("", "_reverse"):
                dirs.append(tuple(
                    w[f"gru.{name}_l{layer}{suffix}"]
                    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
                ))
            self.gru.append(dirs)
        self.out_w = w["out.weight"][0]
        self.out_b = float(w["out.bias"][0])

    def join_probs(self, text: str) -> np.ndarray:
        """Вероятность «склеить» для каждого символа нормализованной строки
        (смысл имеет только у пробелов)."""
        if not text:
            return np.zeros(0, dtype=np.float32)
        x = self.emb[[self.stoi.get(ch, UNK_ID) for ch in text]]
        for fwd, bwd in self.gru:
            x = np.concatenate([
                gru_pass(x, *fwd, self.hid),
                gru_pass(x, *bwd, self.hid, reverse=True),
            ], axis=1)
        return 0.5 * (np.tanh(0.5 * (x @ self.out_w + self.out_b)) + 1.0)

    def segment(self, text: str) -> List[List[str]]:
        """'oron du takil' -> [['oron', 'du'], ['takil']]: группы кусков,
        каждая группа — одно кириллическое слово (или знак препинания)."""
        text = normalize(text)
        if not text:
            return []
        probs = self.join_probs(text)
        groups, cur = [], [""]
        for i, ch in enumerate(text):
            if ch != " ":
                cur[-1] += ch
            elif probs[i] > self.threshold:
                cur.append("")
            else:
                groups.append(cur)
                cur = [""]
        groups.append(cur)
        return groups


_MODEL = None
_LOCK = threading.Lock()


def model_path() -> str:
    return os.environ.get("SEGMENTER_MODEL_PATH") or str(_DEFAULT_MODEL)


def get_model() -> Segmenter:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _MODEL = Segmenter(model_path())
    return _MODEL


def segment(text: str) -> List[List[str]]:
    return get_model().segment(text)
