"""Senim P4: speed of the CTranslate2 backend (senim_fast) against the Hugging Face model (senim).

Translates N FLORES+ dev sentences eng_Latn -> kaz_Cyrl with both backends and prints, per model
and backend: total seconds, seconds per sentence, corpus chrF against the reference, and the share
of outputs identical between the two backends. Ends with one line `SPEED_JSON {...}`.

    pip install ctranslate2 sacrebleu datasets transformers sentencepiece torch
    python experiments/p4_speed_backend.py --n 300                     # GPU if present
    python experiments/p4_speed_backend.py --n 100 --device cpu        # both backends on CPU
    python experiments/p4_speed_backend.py --models 600M,1.3B
Needs the HF_TOKEN environment variable (FLORES+ is gated).
"""
import argparse
import json
import os
import platform
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=300, help="number of FLORES+ dev sentences")
p.add_argument("--device", default="auto", choices=["auto", "cpu"])
p.add_argument("--models", default="600M", help="600M or 600M,1.3B")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--num-beams", type=int, default=4)
args = p.parse_args()

if args.device == "cpu":
    # Hide GPUs before torch / ctranslate2 are imported: senim picks cuda via torch.cuda.is_available(),
    # senim_fast via ctranslate2.get_cuda_device_count().
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SENIM_CT2_DEVICE"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sacrebleu  # noqa: E402
from datasets import load_dataset  # noqa: E402

import senim  # noqa: E402
import senim_fast  # noqa: E402

MODELS = {"600M": "facebook/nllb-200-distilled-600M", "1.3B": "facebook/nllb-200-distilled-1.3B"}
SRC, TGT = "eng_Latn", "kaz_Cyrl"

# ---------------- data ----------------
token = os.environ.get("HF_TOKEN")
eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="dev", token=token)
kaz = load_dataset("openlanguagedata/flores_plus", "kaz_Cyrl", split="dev", token=token)
ref_by_id = dict(zip(kaz["id"], kaz["text"]))
pairs = [(t, ref_by_id[i]) for i, t in zip(eng["id"], eng["text"]) if i in ref_by_id][: args.n]
src = [s for s, _ in pairs]
refs = [r for _, r in pairs]
print(f"{len(src)} sentences, device={args.device}, models={args.models}", flush=True)


def sync_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed(fn, name):
    fn([src[0]], SRC, TGT, model_name=name)  # warm-up: load model, convert once, CUDA init
    sync_cuda()
    t0 = time.perf_counter()
    out = fn(src, SRC, TGT, batch_size=args.batch_size, num_beams=args.num_beams, model_name=name)
    sync_cuda()
    return out, time.perf_counter() - t0


def chrf(hyps):
    return round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)


summary = {"n": len(src), "device_arg": args.device, "batch_size": args.batch_size,
           "num_beams": args.num_beams, "python": platform.python_version(), "models": {}}
try:
    import ctranslate2

    summary["ctranslate2"] = ctranslate2.__version__
    summary["ct2_cuda_devices"] = ctranslate2.get_cuda_device_count()
except Exception as e:  # noqa: BLE001
    summary["ctranslate2"] = f"unavailable: {e}"

for key in args.models.split(","):
    key = key.strip()
    name = MODELS[key]
    t_load = time.perf_counter()
    senim_fast.ensure_converted(name)
    convert_s = round(time.perf_counter() - t_load, 1)

    hf_out, hf_s = timed(senim.translate, name)
    hf_device = senim.load_model(name)[2]
    senim._models.pop(name, None)  # free GPU / RAM before loading the CT2 model
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    ct_out, ct_s = timed(senim_fast.translate, name)
    ct_device = senim_fast.load_model(name)[2]

    same = sum(a.strip() == b.strip() for a, b in zip(hf_out, ct_out)) / len(src)
    res = {
        "hf": {"device": hf_device, "seconds": round(hf_s, 2), "sec_per_sent": round(hf_s / len(src), 4),
               "chrf": chrf(hf_out)},
        "ct2": {"device": ct_device, "seconds": round(ct_s, 2), "sec_per_sent": round(ct_s / len(src), 4),
                "chrf": chrf(ct_out), "convert_seconds_first_call": convert_s},
        "speedup": round(hf_s / ct_s, 2) if ct_s else None,
        "identical_share": round(same, 4),
    }
    summary["models"][key] = res
    for b in ("hf", "ct2"):
        r = res[b]
        print(f"[{key}] {b:4s} device={r['device']:4s} total={r['seconds']:8.2f}s "
              f"per_sent={r['sec_per_sent']:.4f}s chrF={r['chrf']:.2f}", flush=True)
    print(f"[{key}] speedup={res['speedup']}x identical={res['identical_share']:.1%}", flush=True)
    for a, b, s in list(zip(hf_out, ct_out, src))[:3]:
        if a != b:
            print(f"  diff example\n    EN : {s}\n    HF : {a}\n    CT2: {b}")

print("SPEED_JSON " + json.dumps(summary, ensure_ascii=False))
