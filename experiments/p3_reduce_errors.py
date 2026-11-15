"""Senim v3 experiment: can selection among candidate translations reduce errors?

See experiments/P3_PLAN.md (written before running). Run on a GPU (Kaggle or Colab):
    python experiments/p3_reduce_errors.py --split dev            # development, all 997 FLORES+ dev pairs
    python experiments/p3_reduce_errors.py --split devtest --systems A,R   # final run on the notebook's 150 sentences
Needs the HF_TOKEN environment variable (FLORES+ is gated).
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
p.add_argument("--systems", default="A,S,B,R600,R")
p.add_argument("--limit", type=int, default=0, help="use only the first N pairs (for a quick test)")
p.add_argument("--out", default="results")
args = p.parse_args()
systems = args.systems.split(",")
os.makedirs(args.out, exist_ok=True)
tag = f"v3_{args.split}"

# ---------------- data ----------------
from datasets import load_dataset  # noqa: E402

tok = os.environ.get("HF_TOKEN")
eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=args.split, token=tok)
kaz = load_dataset("openlanguagedata/flores_plus", "kaz_Cyrl", split=args.split, token=tok)
df = pd.DataFrame({"id": eng["id"], "en": eng["text"]}).merge(pd.DataFrame({"id": kaz["id"], "ref": kaz["text"]}), on="id")
if args.split == "devtest":  # the same 150 sentences as notebook.ipynb (seed 42 over the 1012 devtest rows)
    idx = np.sort(np.random.default_rng(42).choice(len(df), size=150, replace=False))
    df = df.iloc[idx].reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
src, refs = df["en"].tolist(), df["ref"].tolist()
print(f"[{tag}] pairs: {len(df)}", flush=True)

rated = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "ratings.csv"), dtype=str).fillna("")
rated = rated[rated["human_correction"].str.strip() != ""]
r_src, r_ref = rated["source"].tolist(), rated["human_correction"].tolist()

# ---------------- systems ----------------
outputs, timing, chosen_from = {}, {}, {}
back_cache = {}


def run(name, fn):
    t0 = time.time()
    outputs[name] = fn()
    timing[name] = time.time() - t0
    print(f"[{tag}] {name}: {timing[name]:.0f}s", flush=True)


both = src + r_src  # translate FLORES and the rated sentences together
n = len(src)
cand600 = cand13 = None

if "A" in systems or "R600" in systems or "R" in systems:
    t0 = time.time()
    cand600 = senim.translate_n(both, "eng_Latn", "kaz_Cyrl", n=4, num_beams=4)
    t_c600 = time.time() - t0
    outputs["A"] = [c[0] for c in cand600]
    timing["A"] = t_c600  # upper bound: n-best costs about the same as beam 4
    print(f"[{tag}] A / 600M 4-best: {t_c600:.0f}s", flush=True)
if "S" in systems:
    run("S", lambda: senim.translate_split(both, "eng_Latn", "kaz_Cyrl"))
if "B" in systems or "R" in systems:
    t0 = time.time()
    cand13 = senim.translate_n(both, "eng_Latn", "kaz_Cyrl", n=4, num_beams=4, model_name=senim.BIG_MODEL_NAME)
    timing["B"] = time.time() - t0
    outputs["B"] = [c[0] for c in cand13]
    print(f"[{tag}] B / 1.3B 4-best: {timing['B']:.0f}s", flush=True)
if "R600" in systems:
    t0 = time.time()
    outputs["R600"], chosen_from["R600"], back_cache = senim.select_best(both, cand600, back_cache)
    timing["R600"] = timing["A"] + time.time() - t0
    print(f"[{tag}] R600 selection: {time.time() - t0:.0f}s", flush=True)
if "R" in systems:
    t0 = time.time()
    cands = [a + b for a, b in zip(cand600, cand13)]
    outputs["R"], chosen_from["R"], back_cache = senim.select_best(both, cands, back_cache)
    timing["R"] = timing["A"] + timing["B"] + time.time() - t0
    print(f"[{tag}] R selection: {time.time() - t0:.0f}s", flush=True)

# ---------------- scoring ----------------
rng = np.random.default_rng(0)
boot = rng.integers(0, n, size=(1000, n))


def sent_scores(hyps, rs):
    return np.array([senim.sentence_chrf(h, r) for h, r in zip(hyps, rs)])


base = sent_scores(outputs["A"][:n], refs) if "A" in outputs else None
rows = []
for name in [s for s in systems if s in outputs]:
    hyp, hyp_r = outputs[name][:n], outputs[name][n:]
    ss = sent_scores(hyp, refs)
    row = {"system": name, "split": args.split, "n": n,
           "corpus_chrF": round(sacrebleu.corpus_chrf(hyp, [refs]).score, 2),
           "hard_check_failures": int(sum(bool(senim.hard_checks(s_, h)) for s_, h in zip(src, hyp))),
           "negation_failures": int(sum(bool(senim.check_negation(s_, h)) for s_, h in zip(src, hyp))),
           "rated90_chrF_vs_author_corrections": round(sacrebleu.corpus_chrf(hyp_r, [r_ref]).score, 2),
           "seconds_total": round(timing[name], 1),
           "seconds_per_sentence": round(timing[name] / len(both), 3)}
    if base is not None and name != "A":
        d = ss - base
        bd = d[boot].mean(axis=1)
        row.update({"mean_sent_chrF_gain_vs_A": round(float(d.mean()), 2),
                    "gain_ci95_low": round(float(np.percentile(bd, 2.5)), 2),
                    "gain_ci95_high": round(float(np.percentile(bd, 97.5)), 2)})
    if name in chosen_from:
        ch = np.array(chosen_from[name][:n])
        row["share_not_default"] = round(float((ch != 0).mean()), 3)
        if name == "R":
            row["share_from_1p3B"] = round(float((ch >= 4).mean()), 3)
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(args.out, f"{tag}_summary.csv"), index=False)
per = pd.DataFrame({"id": df["id"], "en": src, "ref": refs, **{k: v[:n] for k, v in outputs.items()}})
per.to_csv(os.path.join(args.out, f"{tag}_translations.csv"), index=False)
ref_neg = int(sum(bool(senim.check_negation(s_, r)) for s_, r in zip(src, refs)))
ref_hard = int(sum(bool(senim.hard_checks(s_, r)) for s_, r in zip(src, refs)))
print(f"===== {tag}_summary.csv")
print(summary.to_csv(index=False))
print(f"references: hard-check failures {ref_hard}, negation failures {ref_neg} (floor for comparison)")
print("RESULT_JSON " + json.dumps({"tag": tag, "rows": rows, "ref_hard": ref_hard, "ref_neg": ref_neg}))
