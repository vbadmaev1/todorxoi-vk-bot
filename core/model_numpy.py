# -*- coding: utf-8 -*-
"""
model_numpy.py — та же модель кириллица→транслитерация, но на чистом
NumPy. Сеть маленькая (1.5 млн параметров), инференс идёт по одному
слову, поэтому весь torch здесь сводится к десятку матричных умножений.

ЗАЧЕМ. Раньше ради этих умножений в образ приезжал torch: 1.2 ГБ сам
плюс 3.2 ГБ пакетов nvidia-*, потому что pip по умолчанию ставит сборку
с CUDA. Пять гигабайт ради двухмегабайтного чекпойнта — и сборка на
хостинге рано или поздно упирается в лимит. С NumPy те же веса весят
3 МБ (float16), а зависимость — 20 МБ.

АРХИТЕКТУРА (повторяет обученную один в один):

    вход: слово посимвольно
      ↓ Embedding
      ↓ двунаправленная GRU  ──► выходы по всем позициям (для внимания)
      ↓ tanh(Linear)         ──► стартовое состояние декодера
      ↓ на каждом шаге: внимание по выходам энкодера, GRUCell, Linear
    выход: транслитерация посимвольно, жадное декодирование

Формулы GRU взяты из документации torch.nn.GRU дословно, включая то, что
r умножается уже на (W_hn·h + b_hn), а не на h — перепутать легко, а
ошибка будет тихой: сеть продолжит выдавать правдоподобные, но другие
слова. Совпадение с исходной реализацией проверяется на всём словаре
скриптом tools/check_numpy_matches_torch.py.

Веса лежат в одном .npz: массивы float16 (или float32) плюс json со
словарями.
"""

import json
import threading
import unicodedata
from pathlib import Path

import numpy as np

PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
META_KEY = "meta_json"


def _sigmoid(x):
    # через tanh — устойчиво к переполнению и без лишних веток
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _gru_step(x, h, w_ih, w_hh, b_ih, b_hh, hid):
    """Один шаг GRU. Порядок ворот в весах torch: r, z, n."""
    return _gru_gates(w_ih @ x + b_ih, h, w_hh, b_hh, hid)


def _gru_gates(gi, h, w_hh, b_hh, hid):
    """Шаг GRU по уже посчитанной входной части gi = W_ih·x + b_ih."""
    gh = w_hh @ h + b_hh
    r = _sigmoid(gi[:hid] + gh[:hid])
    z = _sigmoid(gi[hid:2 * hid] + gh[hid:2 * hid])
    n = np.tanh(gi[2 * hid:] + r * gh[2 * hid:])
    return (1.0 - z) * n + z * h


def gru_pass(xs, w_ih, w_hh, b_ih, b_hh, hid, reverse=False):
    """Один проход GRU по всей последовательности xs (S, in) -> (S, hid).

    Входная часть W_ih·x от прошлых состояний не зависит, поэтому
    считается одним умножением на всю строку; в цикле остаётся только
    W_hh·h. На строке в 40 символов это втрое быстрее пошагового."""
    gi_all = xs @ w_ih.T + b_ih
    out = np.empty((len(xs), hid), dtype=np.float32)
    h = np.zeros(hid, dtype=np.float32)
    steps = range(len(xs) - 1, -1, -1) if reverse else range(len(xs))
    for t in steps:
        h = _gru_gates(gi_all[t], h, w_hh, b_hh, hid)
        out[t] = h
    return out


class CharVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, word):
        unk = self.stoi[UNK]
        return [self.stoi[SOS]] + [self.stoi.get(c, unk) for c in word] + [self.stoi[EOS]]

    def decode(self, ids):
        out = []
        for i in ids:
            ch = self.itos[i]
            if ch == EOS:
                break
            if ch in (PAD, SOS):
                continue
            out.append(ch)
        return "".join(out)


class TranslitModel:
    """Загруженная модель. Потокобезопасна: состояния не хранит, только
    веса, а они только читаются."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(bytes(data[META_KEY]).decode("utf-8"))

        self.src_vocab = CharVocab(meta["src_vocab"])
        self.tgt_vocab = CharVocab(meta["tgt_vocab"])
        self.dictionary = meta["dictionary"]
        self.emb_dim = meta["emb_dim"]
        self.hid = meta["hid_dim"]
        self.max_len = meta["max_len"]
        self.path = str(npz_path)

        # веса могут храниться в float16 (так файл вдвое меньше); считаем
        # всё равно во float32 — приводим один раз при загрузке
        w = {k: data[k].astype(np.float32) for k in data.files if k != META_KEY}
        # энкодер
        self.enc_emb = w["encoder.emb.weight"]
        self.e_wih = w["encoder.gru.weight_ih_l0"]
        self.e_whh = w["encoder.gru.weight_hh_l0"]
        self.e_bih = w["encoder.gru.bias_ih_l0"]
        self.e_bhh = w["encoder.gru.bias_hh_l0"]
        self.e_wih_r = w["encoder.gru.weight_ih_l0_reverse"]
        self.e_whh_r = w["encoder.gru.weight_hh_l0_reverse"]
        self.e_bih_r = w["encoder.gru.bias_ih_l0_reverse"]
        self.e_bhh_r = w["encoder.gru.bias_hh_l0_reverse"]
        self.fc_w = w["encoder.fc.weight"]
        self.fc_b = w["encoder.fc.bias"]
        # декодер
        self.dec_emb = w["decoder.emb.weight"]
        self.attn_w = w["decoder.attention.attn.weight"]
        self.attn_b = w["decoder.attention.attn.bias"]
        # внимание: W·[h_dec ; enc_t] = W_h·h_dec + W_e·enc_t. Вторая часть
        # от шага декодера не зависит — считается один раз на вход (_encode)
        self.attn_wh = np.ascontiguousarray(self.attn_w[:, :self.hid])
        self.attn_we = np.ascontiguousarray(self.attn_w[:, self.hid:])
        self.v_w = w["decoder.attention.v.weight"]          # (1, hid), без bias
        self.d_wih = w["decoder.gru.weight_ih"]
        self.d_whh = w["decoder.gru.weight_hh"]
        self.d_bih = w["decoder.gru.bias_ih"]
        self.d_bhh = w["decoder.gru.bias_hh"]
        self.out_w = w["decoder.out.weight"]
        self.out_b = w["decoder.out.bias"]

        self.sos = self.tgt_vocab.stoi[SOS]
        self.eos = self.tgt_vocab.stoi[EOS]

    # ------------------------------------------------------------ энкодер

    def _encode(self, ids):
        hid = self.hid
        embs = self.enc_emb[ids]                       # (S, emb)
        fwd = gru_pass(embs, self.e_wih, self.e_whh, self.e_bih, self.e_bhh, hid)
        bwd = gru_pass(embs, self.e_wih_r, self.e_whh_r, self.e_bih_r, self.e_bhh_r,
                       hid, reverse=True)
        # последнее состояние прямого прохода — на последнем символе,
        # обратного — на первом
        enc_outputs = np.concatenate([fwd, bwd], axis=1)          # (S, 2*hid)
        h0 = np.tanh(self.fc_w @ np.concatenate([fwd[-1], bwd[0]]) + self.fc_b)
        return enc_outputs, h0

    # ------------------------------------------------------------ внимание

    def _attention(self, h_dec, enc_proj):
        # energy = tanh(W · [h_dec ; enc_out_t] + b) для каждой позиции t;
        # enc_proj = W_e·enc_out + b посчитан заранее
        energy = np.tanh(enc_proj + self.attn_wh @ h_dec)         # (S, hid)
        scores = energy @ self.v_w[0]                             # (S,)
        return _softmax(scores)

    # ------------------------------------------------------------ декодер

    def translate_word(self, word):
        word = unicodedata.normalize("NFC", word)
        ids = self.src_vocab.encode(word)
        enc_outputs, h = self._encode(ids)
        enc_proj = enc_outputs @ self.attn_we.T + self.attn_b     # (S, hid)

        token = self.sos
        out_ids = []
        for _ in range(self.max_len):
            emb = self.dec_emb[token]
            attn = self._attention(h, enc_proj)
            context = attn @ enc_outputs                          # (2*hid,)
            h = _gru_step(
                np.concatenate([emb, context]), h,
                self.d_wih, self.d_whh, self.d_bih, self.d_bhh, self.hid,
            )
            logits = self.out_w @ np.concatenate([h, context, emb]) + self.out_b
            token = int(logits.argmax())
            if token == self.eos:
                break
            out_ids.append(token)
        return self.tgt_vocab.decode(out_ids)


# ---------------------------------------------------------------- загрузка

_MODELS = {}
_LOCK = threading.Lock()


def load(npz_path) -> TranslitModel:
    """Ленивая загрузка: один экземпляр на файл и процесс. Файлов два —
    кириллица→транслитерация и обратная модель (core/to_cyrillic.py)."""
    key = str(npz_path)
    if key not in _MODELS:
        with _LOCK:
            if key not in _MODELS:
                path = Path(npz_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Не найден файл модели: {path}. Он лежит в репозитории "
                        f"в папке model/; если собираете сами — "
                        f"python -m tools.export_model_npz <чекпойнт.pt>"
                    )
                _MODELS[key] = TranslitModel(path)
    return _MODELS[key]
