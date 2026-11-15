"""Tests for senim_fast (CTranslate2 backend). No models, ctranslate2 or transformers needed:
the translator, tokenizer and converter are replaced by small fakes.

Run:  python -m pytest tests/
"""
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import senim_fast as sf  # noqa: E402


# ---------------------------------------------------------------- fakes
class FakeTok:
    """Whitespace tokenizer that mimics the NLLB fast tokenizer API used by senim_fast."""

    SPECIAL = {"</s>", "eng_Latn", "kaz_Cyrl"}

    def __init__(self):
        self.src_lang = None
        self.vocab = {}
        self.encoded_with = []

    def _id(self, t):
        return self.vocab.setdefault(t, len(self.vocab))

    def encode(self, text, truncation=False, max_length=None):
        self.encoded_with.append(self.src_lang)
        toks = [self.src_lang] + text.split() + ["</s>"]
        if truncation and max_length and len(toks) > max_length:
            toks = toks[:max_length - 1] + ["</s>"]
        return [self._id(t) for t in toks]

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv[i] for i in ids]

    def convert_tokens_to_ids(self, toks):
        return [self._id(t) for t in toks]

    def decode(self, ids, skip_special_tokens=False):
        toks = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            toks = [t for t in toks if t not in self.SPECIAL]
        return " ".join(toks)


class FakeResult:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class FakeTranslator:
    """'Translates' by upper-casing source words; hypothesis k gets suffix -k."""

    def __init__(self):
        self.calls = []

    def translate_batch(self, sources, **kw):
        self.calls.append((sources, kw))
        tgt = kw["target_prefix"]
        out = []
        for src, prefix in zip(sources, tgt):
            words = [w.upper() for w in src[1:-1]]  # drop src lang token and </s>
            hyps = [prefix + [f"{w}-{k}" if k else w for w in words]
                    for k in range(kw.get("num_hypotheses", 1))]
            out.append(FakeResult(hyps))
        return out


@pytest.fixture
def fake_backend(monkeypatch):
    tr, tok = FakeTranslator(), FakeTok()
    monkeypatch.setattr(sf, "load_model", lambda name=sf.MODEL_NAME, device=None: (tr, tok, "cpu"))
    return tr, tok


# ---------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("name,expected", [
    ("facebook/nllb-200-distilled-600M", "facebook--nllb-200-distilled-600M"),
    ("facebook/nllb-200-distilled-1.3B", "facebook--nllb-200-distilled-1.3B"),
    ("C:\\models\\my model", "C_--models--my_model"),
    ("/abs/path/", "abs--path"),
    ("..", "model"),
])
def test_sanitize_name(name, expected):
    out = sf.sanitize_name(name)
    assert out == expected
    assert "/" not in out and "\\" not in out and ":" not in out


def test_sanitize_name_distinguishes_models():
    assert sf.sanitize_name(sf.MODEL_NAME) != sf.sanitize_name(sf.BIG_MODEL_NAME)


def test_pick_device():
    assert sf.pick_device(1) == ("cuda", "int8_float16")
    assert sf.pick_device(0) == ("cpu", "int8")
    assert sf.pick_device(2, "cpu") == ("cpu", "int8")
    assert sf.pick_device(0, " CUDA ") == ("cuda", "int8_float16")
    with pytest.raises(ValueError):
        sf.pick_device(0, "tpu")


def test_order_roundtrip():
    texts = ["ccc", "a", "bb", "", "dddd", "a"]
    order = sf.length_sorted_order(texts)
    assert [len(texts[i]) for i in order] == sorted(len(t) for t in texts)
    assert order.index(1) < order.index(5)  # stable for equal lengths
    assert sf.restore_order(order, [texts[i] for i in order]) == texts


def test_restore_order_length_mismatch():
    with pytest.raises(ValueError):
        sf.restore_order([0, 1], ["x"])


def test_strip_lang_token():
    assert sf.strip_lang_token(["kaz_Cyrl", "▁Сә", "лем"], "kaz_Cyrl") == ["▁Сә", "лем"]
    assert sf.strip_lang_token(["▁Сә", "лем"], "kaz_Cyrl") == ["▁Сә", "лем"]
    assert sf.strip_lang_token([], "kaz_Cyrl") == []
    # only the leading token is removed
    assert sf.strip_lang_token(["kaz_Cyrl", "kaz_Cyrl"], "kaz_Cyrl") == ["kaz_Cyrl"]


def test_decode_hypothesis_drops_lang_and_special():
    tok = FakeTok()
    assert sf.decode_hypothesis(tok, ["kaz_Cyrl", "сәлем", "әлем", "</s>"], "kaz_Cyrl") == "сәлем әлем"


def test_encode_sources_sets_src_lang_and_truncates():
    tok = FakeTok()
    tok.src_lang = "kaz_Cyrl"
    out = sf.encode_sources(tok, ["hello world", "x " * 500], "eng_Latn", max_tokens=10)
    assert tok.src_lang == "eng_Latn"
    assert set(tok.encoded_with) == {"eng_Latn"}
    assert out[0] == ["eng_Latn", "hello", "world", "</s>"]
    assert len(out[1]) == 10 and out[1][-1] == "</s>"


# ---------------------------------------------------------------- translate / translate_n with fakes
def test_translate_keeps_order_and_params(fake_backend):
    tr, _ = fake_backend
    texts = ["a long sentence here", "hi", "mid size"]
    out = sf.translate(texts, "eng_Latn", "kaz_Cyrl", batch_size=5, num_beams=3, max_new_tokens=77)
    assert out == ["A LONG SENTENCE HERE", "HI", "MID SIZE"]
    sources, kw = tr.calls[-1]
    assert [s[0] for s in sources] == ["eng_Latn"] * 3
    assert [len(s) for s in sources] == sorted(len(s) for s in sources)  # length-sorted
    assert kw == {"target_prefix": [["kaz_Cyrl"]] * 3, "beam_size": 3, "num_hypotheses": 1,
                  "max_decoding_length": 77, "max_batch_size": 5}


def test_translate_accepts_str_and_empty(fake_backend):
    assert sf.translate("hello", "eng_Latn", "kaz_Cyrl") == ["HELLO"]
    assert sf.translate([], "eng_Latn", "kaz_Cyrl") == []
    assert sf.translate_n([], "eng_Latn", "kaz_Cyrl") == []


def test_translate_n_shapes(fake_backend):
    tr, _ = fake_backend
    texts = ["bb cc", "a"]
    out = sf.translate_n(texts, "eng_Latn", "kaz_Cyrl", n=3, num_beams=2)
    assert out == [["BB CC", "BB-1 CC-1", "BB-2 CC-2"], ["A", "A-1", "A-2"]]
    _, kw = tr.calls[-1]
    assert kw["beam_size"] == 3 and kw["num_hypotheses"] == 3 and kw["max_batch_size"] == 8
    assert out[0][0] == sf.translate(texts, "eng_Latn", "kaz_Cyrl")[0]


# ---------------------------------------------------------------- conversion with a fake converter
def test_ensure_converted_converts_once(tmp_path, monkeypatch):
    calls = []

    class FakeConverter:
        def __init__(self, name):
            self.name = name

        def convert(self, out, quantization=None, force=False):
            calls.append((self.name, quantization))
            Path(out, "model.bin").write_bytes(b"x")
            Path(out, "config.json").write_text("{}")
            return out

    ct2 = types.ModuleType("ctranslate2")
    conv = types.ModuleType("ctranslate2.converters")
    conv.TransformersConverter = FakeConverter
    ct2.converters = conv
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", conv)
    monkeypatch.setenv("SENIM_CT2_CACHE", str(tmp_path))

    p1 = sf.ensure_converted("facebook/nllb-200-distilled-600M")
    p2 = sf.ensure_converted("facebook/nllb-200-distilled-600M")
    assert p1 == p2 == str(tmp_path / "facebook--nllb-200-distilled-600M")
    assert sf.is_converted(p1)
    assert calls == [("facebook/nllb-200-distilled-600M", "int8")]
    assert [p.name for p in tmp_path.iterdir()] == ["facebook--nllb-200-distilled-600M"]  # no temp dirs left

    custom = tmp_path / "custom"
    assert sf.ensure_converted("m", out_dir=custom, quantization="int8_float16") == str(custom)
    assert calls[-1] == ("m", "int8_float16")


def test_real_ctranslate2_api_present():
    ct2 = pytest.importorskip("ctranslate2")
    from ctranslate2.converters import TransformersConverter  # noqa: F401

    assert hasattr(ct2, "Translator") and hasattr(ct2, "get_cuda_device_count")
