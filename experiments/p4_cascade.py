"""Senim v4 experiment: is the cascade (best-of-8 only for risky sentences) as good as v3 but faster?

See experiments/P4_PLAN.md (written before running). Run on a GPU:
    python experiments/p4_cascade.py --split dev
    python experiments/p4_cascade.py --split devtest --systems A,C
Needs HF_TOKEN (FLORES+ is gated).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import senim  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", default="dev", choices=["dev", "devtest"])
p.add_argument("--systems", default="A,R,C")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--threshold", type=float, default=63.0)
p.add_argument("--out", default="results")
args = p.parse_args()
systems = args.systems.split(",")
os.makedirs(args.out, exist_ok=True)
tag = f"v4_{args.split}"

from datasets import load_dataset  # noqa: E402

tok = os.environ.get("HF_TOKEN")
eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=args.split, token=tok)
kaz = load_dataset("openlanguagedata/flores_plus", "kaz_Cyrl", split=args.split, token=tok)
df = pd.DataFrame({"id": eng["id"], "en": eng["text"]}).merge(pd.DataFrame({"id": kaz["id"], "ref": kaz["text"]}), on="id")
if args.split == "devtest":
    idx = np.sort(np.random.default_rng(42).choice(len(df), size=150, replace=False))
    df = df.iloc[idx].reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
src, refs = df["en"].tolist(), df["ref"].tolist()
n = len(src)
print(f"[{tag}] pairs: {n}", flush=True)

# warm up both models so that loading time is not counted
senim.translate(["Hello."], "eng_Latn", "kaz_Cyrl")
senim.translate(["Hello."], "eng_Latn", "kaz_Cyrl", model_name=senim.BIG_MODEL_NAME)

import torch  # noqa: E402


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    res = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return res, time.time() - t0


outputs, secs, extra = {}, {}, {}

if "A" in systems:
    # A as the app runs it: translation + back-translation for the trust check
    def run_a():
        out = [c[0] for c in senim.translate_n(src, "eng_Latn", "kaz_Cyrl", n=4, num_beams=4)]
        senim.back_translate_cached(out)
        return out
    outputs["A"], secs["A"] = timed(run_a)
    print(f"[{tag}] A: {secs['A']:.0f}s", flush=True)

if "R" in systems:
    def run_r():
        c600 = senim.translate_n(src, "eng_Latn", "kaz_Cyrl", n=4, num_beams=4)
        c13 = senim.translate_n(src, "eng_Latn", "kaz_Cyrl", n=4, num_beams=4, model_name=senim.BIG_MODEL_NAME)
        chosen, _, _ = senim.select_best(src, [a + b for a, b in zip(c600, c13)])
        return chosen
    outputs["R"], secs["R"] = timed(run_r)
    print(f"[{tag}] R: {secs['R']:.0f}s", flush=True)

if "C" in systems:
    (outC, stages, _), secs["C"] = timed(lambda: senim.smart_translate(src, threshold=args.threshold))
    outputs["C"] = outC
    extra["C_share_best_of_8"] = round(float(np.mean([s == "best_of_8" for s in stages])), 3)
    print(f"[{tag}] C: {secs['C']:.0f}s, best_of_8 share {extra['C_share_best_of_8']}", flush=True)

rng = np.random.default_rng(0)
boot = rng.integers(0, n, size=(1000, n))


def sent(h):
    return np.array([senim.sentence_chrf(a, b) for a, b in zip(h, refs)])


sA = sent(outputs["A"]) if "A" in outputs else None
rows = []
for name in [s for s in systems if s in outputs]:
    hyp = outputs[name]
    row = {"system": name, "split": args.split, "n": n,
           "corpus_chrF": round(sacrebleu.corpus_chrf(hyp, [refs]).score, 2),
           "seconds_total": round(secs[name], 1), "seconds_per_sentence": round(secs[name] / n, 3)}
    if sA is not None and name != "A":
        d = sent(hyp) - sA
        bd = d[boot].mean(axis=1)
        row.update({"mean_sent_chrF_gain_vs_A": round(float(d.mean()), 2),
                    "gain_ci95_low": round(float(np.percentile(bd, 2.5)), 2),
                    "gain_ci95_high": round(float(np.percentile(bd, 97.5)), 2)})
    if name == "C":
        row["share_best_of_8"] = extra["C_share_best_of_8"]
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(args.out, f"{tag}_summary.csv"), index=False)
pd.DataFrame({"id": df["id"], "en": src, "ref": refs, **outputs}).to_csv(
    os.path.join(args.out, f"{tag}_translations.csv"), index=False)
print(f"===== {tag}_summary.csv")
print(summary.to_csv(index=False))
print("RESULT_JSON " + json.dumps({"tag": tag, "rows": rows}))
