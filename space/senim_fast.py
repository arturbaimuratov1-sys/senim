"""Senim fast backend: NLLB-200 translation with CTranslate2.

A drop-in replacement for ``senim.translate`` and ``senim.translate_n`` that runs the
same Hugging Face NLLB-200 checkpoints (``facebook/nllb-200-distilled-600M`` and
``facebook/nllb-200-distilled-1.3B``) through CTranslate2 instead of PyTorch. The
model is converted once to CTranslate2 format (weights stored as int8) and cached in
``~/.cache/senim_ct2/<name>``. At load time it runs with compute_type
``int8_float16`` on a CUDA GPU and ``int8`` on CPU, which is several times faster
than the PyTorch model and uses far less memory (useful on a free 2-vCPU
Hugging Face Space).

Outputs can differ slightly from the Hugging Face fp16/fp32 model: int8 weight
quantisation changes the scores a little, so beam search sometimes picks a
different (usually equally good) hypothesis. Measure the gap with
``experiments/p4_speed_backend.py`` before relying on exact equality.

Recipe follows the CTranslate2 Transformers guide (NLLB section):
tokens = tok.convert_ids_to_tokens(tok.encode(text)) with tok.src_lang set,
translate_batch(..., target_prefix=[[tgt_lang]]), then drop the first output token
(the forced target-language token) and decode.

Environment variables:
  SENIM_CT2_DEVICE   force "cpu" or "cuda" (default: cuda if available)
  SENIM_CT2_CACHE    cache root for converted models (default ~/.cache/senim_ct2)
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MODEL_NAME = "facebook/nllb-200-distilled-600M"
BIG_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MAX_SOURCE_TOKENS = 256  # same input truncation as senim.py

_models: dict[tuple[str, str], tuple[Any, Any, str]] = {}  # (name, device) -> (translator, tok, device)
_load_lock = threading.Lock()
_tok_lock = threading.Lock()  # tok.src_lang is shared mutable state


# ---------------------------------------------------------------------------
# Pure helpers (tested without models)
# ---------------------------------------------------------------------------

def sanitize_name(model_name: str) -> str:
    """Turn a HF model id into a safe directory name: 'facebook/nllb-200' -> 'facebook--nllb-200'."""
    name = model_name.strip().replace("\\", "/").strip("/")
    name = name.replace("/", "--")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip(".") or "model"
    return name


def default_cache_root() -> Path:
    root = os.environ.get("SENIM_CT2_CACHE")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "senim_ct2"


def is_converted(path: str | os.PathLike) -> bool:
    """A CTranslate2 model directory contains model.bin and config.json."""
    p = Path(path)
    return (p / "model.bin").is_file() and (p / "config.json").is_file()


def pick_device(cuda_device_count: int, override: str | None = None) -> tuple[str, str]:
    """Return (device, compute_type)."""
    device = (override or "").strip().lower() or ("cuda" if cuda_device_count > 0 else "cpu")
    if device not in ("cuda", "cpu"):
        raise ValueError(f"unknown device {device!r}; use 'cuda' or 'cpu'")
    return device, ("int8_float16" if device == "cuda" else "int8")


def length_sorted_order(texts: Sequence[str]) -> list[int]:
    """Indices sorted by text length (stable), so batches hold similar lengths."""
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


def restore_order(order: Sequence[int], results: Sequence[Any]) -> list[Any]:
    """results[k] belongs to input index order[k]; return results in input order."""
    if len(order) != len(results):
        raise ValueError("order and results differ in length")
    out: list[Any] = [None] * len(order)
    for i, r in zip(order, results):
        out[i] = r
    return out


def strip_lang_token(hyp: Sequence[str], tgt_lang: str) -> list[str]:
    """Drop the forced target-language token at the start of a hypothesis (if present)."""
    hyp = list(hyp)
    if hyp and hyp[0] == tgt_lang:
        return hyp[1:]
    return hyp


def decode_hypothesis(tok: Any, hyp: Sequence[str], tgt_lang: str) -> str:
    """Tokens from CTranslate2 -> text, matching HF batch_decode(skip_special_tokens=True)."""
    ids = tok.convert_tokens_to_ids(strip_lang_token(hyp, tgt_lang))
    return tok.decode(ids, skip_special_tokens=True)


def encode_sources(tok: Any, texts: Iterable[str], src_lang: str,
                   max_tokens: int = MAX_SOURCE_TOKENS) -> list[list[str]]:
    """Tokenise texts as NLLB source tokens ([src_lang, ..., '</s>']), truncated like senim.py."""
    with _tok_lock:  # set src_lang right before encoding; no other thread can change it meanwhile
        tok.src_lang = src_lang
        return [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=max_tokens))
                for t in texts]


def _as_list(texts: str | Iterable[Any]) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]
    return [str(t) for t in texts]


# ---------------------------------------------------------------------------
# Conversion and loading
# ---------------------------------------------------------------------------

def ensure_converted(model_name: str, out_dir: str | os.PathLike | None = None,
                     quantization: str = "int8") -> str:
    """Convert a HF NLLB model to CTranslate2 format once; return the model directory.

    Skips the work if the directory already holds a converted model. Converts into a
    temporary sibling directory and renames it, so an interrupted run never leaves a
    half-written model that would be mistaken for a finished one.
    """
    out = Path(out_dir) if out_dir else default_cache_root() / sanitize_name(model_name)
    if is_converted(out):
        return str(out)
    from ctranslate2.converters import TransformersConverter

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        TransformersConverter(model_name).convert(str(tmp), quantization=quantization, force=True)
        if is_converted(out):  # another process finished first
            return str(out)
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return str(out)


def load_model(model_name: str = MODEL_NAME, device: str | None = None):
    """Return (translator, tokenizer, device), cached per (model_name, device)."""
    import ctranslate2

    dev, compute_type = pick_device(ctranslate2.get_cuda_device_count(),
                                    device or os.environ.get("SENIM_CT2_DEVICE"))
    key = (model_name, dev)
    if key not in _models:
        with _load_lock:
            if key not in _models:
                from transformers import AutoTokenizer

                path = ensure_converted(model_name)
                translator = ctranslate2.Translator(path, device=dev, compute_type=compute_type)
                tok = AutoTokenizer.from_pretrained(model_name)
                _models[key] = (translator, tok, dev)
    return _models[key]


# ---------------------------------------------------------------------------
# Translation (same signatures as senim.py)
# ---------------------------------------------------------------------------

def _run(texts, src_lang, tgt_lang, n, num_beams, batch_size, max_new_tokens, model_name):
    translator, tok, _ = load_model(model_name)
    texts = _as_list(texts)
    if not texts:
        return []
    order = length_sorted_order(texts)
    sources = encode_sources(tok, [texts[i] for i in order], src_lang)
    results = translator.translate_batch(
        sources,
        target_prefix=[[tgt_lang]] * len(sources),
        beam_size=max(num_beams, n),
        num_hypotheses=n,
        max_decoding_length=max_new_tokens,
        max_batch_size=batch_size,
    )
    decoded = [[decode_hypothesis(tok, h, tgt_lang) for h in r.hypotheses] for r in results]
    return restore_order(order, decoded)


def translate(texts, src_lang: str, tgt_lang: str, batch_size: int = 16,
              num_beams: int = 4, max_new_tokens: int = 256, model_name: str = MODEL_NAME):
    """Translate a list of strings with NLLB via CTranslate2. Codes are FLORES-200, e.g. eng_Latn."""
    return [c[0] for c in _run(texts, src_lang, tgt_lang, 1, num_beams, batch_size,
                               max_new_tokens, model_name)]


def translate_n(texts, src_lang: str, tgt_lang: str, n: int = 4, num_beams: int = 4,
                model_name: str = MODEL_NAME, batch_size: int = 8, max_new_tokens: int = 256):
    """Return the n best beam-search translations for every text (list of lists).
    The first candidate equals translate(...) with the same num_beams (when num_beams >= n)."""
    return _run(texts, src_lang, tgt_lang, n, num_beams, batch_size, max_new_tokens, model_name)
