"""Builds notebook.ipynb from the cell list below (keeps the notebook diff-friendly)."""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# Senim — English → Kazakh translation that warns when not to trust it

Veritas AI Hackathon 2026 · Track AI + Languages · based on brief 4.4.2 *Translation quality test*.

**Run on Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU), then Runtime → Run all.
Needs a Colab secret `HF_TOKEN` (FLORES+ is a gated dataset: accept its terms on Hugging Face first).

Steps:
1. Load 150 FLORES+ devtest sentences (English + Kazakh/Russian/French references).
2. Translate with `facebook/nllb-200-distilled-600M`, score chrF per language, compare with a "leave it in English" baseline.
3. Show the 5 worst Kazakh translations.
4. Translate our own set (`data/own_set.csv`: medicine leaflets and school letters).
5. Trust check: numbers / dosages / dates + back-translation chrF.
6. With quality ratings (`data/ratings.csv`, given by an LLM judge, see below): correlation with chrF, and how many bad translations the trust check catches vs. "always trust".

*This project is a prototype and is not medical advice.*
""")

code("""
# Setup: get the repo (data + shared code) and install the one missing library.
import os, sys, subprocess
REPO_URL = "https://github.com/arturbaimuratov1-sys/senim"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/senim"):
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/senim"], check=True)
    else:
        subprocess.run(["git", "-C", "/content/senim", "pull", "-q"], check=True)
    os.chdir("/content/senim")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu"], check=True)
sys.path.insert(0, os.getcwd())
os.makedirs("results", exist_ok=True)

import torch
print("cwd:", os.getcwd())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU will be slow)")
""")

code("""
# Hugging Face token from the Colab secret HF_TOKEN (never printed or saved).
HF_TOKEN = None
if IN_COLAB:
    from google.colab import userdata
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception as e:
        print("Could not read the HF_TOKEN secret:", type(e).__name__)
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
print("HF token available:", bool(HF_TOKEN))
""")

md("## 1. FLORES+ devtest: 150 sentences")

code("""
import numpy as np, pandas as pd
from datasets import load_dataset

DATASET = "openlanguagedata/flores_plus"
LANGS = {"kaz_Cyrl": "Kazakh", "rus_Cyrl": "Russian", "fra_Latn": "French"}
N = 150
SEED = 42

def load_lang(code):
    ds = load_dataset(DATASET, code, split="devtest", token=HF_TOKEN)
    return pd.DataFrame({"id": ds["id"], code: ds["text"]})

flores = load_lang("eng_Latn")
for code_ in LANGS:
    flores = flores.merge(load_lang(code_), on="id")
print("devtest sentences with all 4 languages:", len(flores))

rng = np.random.default_rng(SEED)
sample_idx = np.sort(rng.choice(len(flores), size=N, replace=False))
flores = flores.iloc[sample_idx].reset_index(drop=True)
flores.head(3)
""")

md("## 2. Translate with NLLB-200 (600M distilled)")

code("""
import time, senim
t0 = time.time()
first = senim.translate([flores.loc[0, "eng_Latn"]], "eng_Latn", "kaz_Cyrl")[0]
print("EN:", flores.loc[0, "eng_Latn"])
print("KK (NLLB):", first)
print("KK (reference):", flores.loc[0, "kaz_Cyrl"])
print(f"model load + first sentence: {time.time() - t0:.1f}s")
""")

code("""
t0 = time.time()
for code_ in LANGS:
    flores[f"mt_{code_}"] = senim.translate(flores["eng_Latn"].tolist(), "eng_Latn", code_)
    print(f"{LANGS[code_]}: done ({time.time() - t0:.0f}s)")
""")

md("""
## 3. chrF per language vs. the "no translation" baseline

Baseline = hand the reader the English text unchanged. chrF counts overlapping character n-grams with the human reference (0–100).
""")

code("""
import sacrebleu
rows = []
for code_, name in LANGS.items():
    refs = flores[code_].tolist()
    mt = sacrebleu.corpus_chrf(flores[f"mt_{code_}"].tolist(), [refs]).score
    base = sacrebleu.corpus_chrf(flores["eng_Latn"].tolist(), [refs]).score
    rows.append({"language": name, "code": code_, "n": len(refs),
                 "chrF_NLLB": round(mt, 1), "chrF_baseline_english": round(base, 1)})
chrf_table = pd.DataFrame(rows)
chrf_table.to_csv("results/chrf_by_language.csv", index=False)
chrf_table
""")

code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(chrf_table))
ax.bar(x - 0.2, chrf_table["chrF_NLLB"], 0.4, label="NLLB-200 600M", color="#2a6f97")
ax.bar(x + 0.2, chrf_table["chrF_baseline_english"], 0.4, label="baseline: leave English", color="#bbbbbb")
for i, r in chrf_table.iterrows():
    ax.text(i - 0.2, r["chrF_NLLB"] + 1, f'{r["chrF_NLLB"]:.1f}', ha="center", fontsize=9)
    ax.text(i + 0.2, r["chrF_baseline_english"] + 1, f'{r["chrF_baseline_english"]:.1f}', ha="center", fontsize=9)
ax.set_xticks(x, chrf_table["language"])
ax.set_ylabel("corpus chrF (FLORES+ devtest, n=150)")
ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/chrf_by_language.png", dpi=150)
plt.show()
""")

md("## 4. Five worst Kazakh translations (sentence chrF)")

code("""
flores["chrf_kaz"] = [senim.sentence_chrf(h, r) for h, r in zip(flores["mt_kaz_Cyrl"], flores["kaz_Cyrl"])]
worst = flores.nsmallest(5, "chrf_kaz")[["id", "eng_Latn", "kaz_Cyrl", "mt_kaz_Cyrl", "chrf_kaz"]]
worst = worst.rename(columns={"eng_Latn": "source_en", "kaz_Cyrl": "reference_kk", "mt_kaz_Cyrl": "nllb_kk"})
worst.to_csv("results/worst5_kaz.csv", index=False)
with pd.option_context("display.max_colwidth", None):
    display(worst)
""")

md("""
## 5. Back-translation on FLORES (no human needed)

Translate NLLB's Kazakh back to English and compare with the English source (chrF).
If this reference-free score tracks the reference-based score, it is a usable warning signal when no reference exists.
""")

code("""
from scipy.stats import spearmanr, pearsonr
flores["back_en"] = senim.translate(flores["mt_kaz_Cyrl"].tolist(), "kaz_Cyrl", "eng_Latn")
flores["bt_chrf"] = [senim.sentence_chrf(b, s) for b, s in zip(flores["back_en"], flores["eng_Latn"])]
flores["hard_check_reasons"] = ["; ".join(senim.hard_checks(s, t)) for s, t in zip(flores["eng_Latn"], flores["mt_kaz_Cyrl"])]
rho, p = spearmanr(flores["bt_chrf"], flores["chrf_kaz"])
print(f"Spearman(back-translation chrF, reference chrF) on {len(flores)} FLORES sentences: rho={rho:.2f}, p={p:.1e}")
print("FLORES sentences flagged by hard checks:", (flores["hard_check_reasons"] != "").sum())
flores.to_csv("results/flores_translations.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(flores["bt_chrf"], flores["chrf_kaz"], s=14, alpha=0.7, color="#2a6f97")
ax.set_xlabel("back-translation chrF (vs English source)")
ax.set_ylabel("chrF vs Kazakh reference")
ax.set_title(f"FLORES+ kaz, n={len(flores)}, Spearman rho={rho:.2f}", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/bt_vs_reference_chrf.png", dpi=150); plt.show()
""")

md("""
## 6. Our own set: medicine leaflets and school letters

`data/own_set.csv` was written by the project author (columns `id, text_en, type`).
""")

code("""
own = pd.read_csv("data/own_set.csv", dtype=str).fillna("")
own = own[own["text_en"].str.strip() != ""].reset_index(drop=True)
print("own sentences:", len(own), dict(own["type"].value_counts()) if len(own) else {})
if len(own):
    own["mt_kaz"] = senim.translate(own["text_en"].tolist(), "eng_Latn", "kaz_Cyrl")
    own["back_en"] = senim.translate(own["mt_kaz"].tolist(), "kaz_Cyrl", "eng_Latn")
    own["bt_chrf"] = [senim.sentence_chrf(b, s) for b, s in zip(own["back_en"], own["text_en"])]
    own["hard_check_reasons"] = ["; ".join(senim.hard_checks(s, t)) for s, t in zip(own["text_en"], own["mt_kaz"])]
    own.to_csv("results/own_set_translations.csv", index=False)
    with pd.option_context("display.max_colwidth", 80):
        display(own[["id", "type", "text_en", "mt_kaz", "bt_chrf", "hard_check_reasons"]].head(10))
else:
    print("data/own_set.csv is empty -> skipped (fill it in and re-run).")
""")

md("""
## 7. Rating sheet

~90 translations (all own sentences + 40 random FLORES ones), shuffled, **without** chrF or trust labels so the rater is not biased.
Written to `results/ratings_template.csv` only if `data/ratings.csv` has no ratings yet.
""")

code("""
def ratings_filled():
    if not os.path.exists("data/ratings.csv"):
        return False
    r = pd.read_csv("data/ratings.csv", dtype=str).fillna("")
    return len(r) > 0 and (r["rating_1_5"].str.strip() != "").any()

flores_rate = flores.sample(n=40, random_state=7)
pool = pd.concat([
    pd.DataFrame({"id": "own_" + own["id"].astype(str), "source": own.get("text_en", []), "translation": own.get("mt_kaz", [])}) if len(own) else pd.DataFrame(),
    pd.DataFrame({"id": "flores_" + flores_rate["id"].astype(str), "source": flores_rate["eng_Latn"], "translation": flores_rate["mt_kaz_Cyrl"]}),
], ignore_index=True)
if not ratings_filled() and len(own):
    sheet = pool.sample(frac=1, random_state=3).reset_index(drop=True)
    sheet["rating_1_5"] = ""; sheet["error_type"] = ""
    sheet.to_csv("results/ratings_template.csv", index=False)
    print("wrote results/ratings_template.csv with", len(sheet), "rows")
else:
    print("ratings already filled or own set empty -> template not written")
""")

md("""
## 8. Evaluation against quality ratings

**Who rated:** the 1-5 ratings in `data/ratings.csv` were given by **Claude Opus 5.5 acting as an LLM judge** (column `rater`), not by a fluent human rater.
The author read the translations and judged them as "meaning mostly conveyed, but not exact, and some Russian words appear", but did not rate them one by one.
Treat every number below as agreement with an LLM judge, not with people.

**Pre-registered rules (fixed before seeing ratings):**
- A translation is **bad** if `rating_1_5 <= 3` (meaning partly wrong or worse — not safe for a medicine leaflet). We also report `<= 2`.
- Ratings are split 50/50 at random (seed 2026) into a **calibration** part and a **test** part.
  The back-translation threshold is chosen on calibration only (maximise F1 for catching bad translations); all reported trust numbers are on the test part.
- Baseline: **always trust** (never warns).
""")

code("""
import json
if not ratings_filled():
    print("data/ratings.csv has no ratings yet -> evaluation skipped.")
else:
    ratings = pd.read_csv("data/ratings.csv", dtype=str).fillna("")
    ratings = ratings[ratings["rating_1_5"].str.strip() != ""].copy()
    ratings["rating"] = ratings["rating_1_5"].astype(float)
    feats = pd.concat([
        pd.DataFrame({"id": "own_" + own["id"].astype(str), "bt_chrf": own["bt_chrf"], "hard": own["hard_check_reasons"], "ref_chrf": np.nan, "src_type": own["type"]}) if len(own) else pd.DataFrame(),
        pd.DataFrame({"id": "flores_" + flores["id"].astype(str), "bt_chrf": flores["bt_chrf"], "hard": flores["hard_check_reasons"], "ref_chrf": flores["chrf_kaz"], "src_type": "flores"}),
    ], ignore_index=True)
    ev = ratings.merge(feats, on="id", how="left")
    print("rated translations:", len(ev), "| by source:", dict(ev["src_type"].value_counts()))
    print("rating distribution:", dict(ev["rating"].value_counts().sort_index()))
    print("error types:", dict(ev["error_type"].replace("", "(none)").value_counts()))
""")

code("""
if ratings_filled():
    corr_rows = []
    fl = ev.dropna(subset=["ref_chrf"])
    if len(fl) >= 5:
        r_s = spearmanr(fl["rating"], fl["ref_chrf"]); r_p = pearsonr(fl["rating"], fl["ref_chrf"])
        corr_rows.append({"metric": "chrF vs Kazakh reference", "subset": "FLORES rated", "n": len(fl),
                          "spearman": round(r_s[0], 3), "spearman_p": r_s[1], "pearson": round(r_p[0], 3)})
    for name, sub in [("all rated", ev), ("own set", ev[ev["src_type"] != "flores"]), ("FLORES rated", fl)]:
        if len(sub) >= 5:
            r_s = spearmanr(sub["rating"], sub["bt_chrf"]); r_p = pearsonr(sub["rating"], sub["bt_chrf"])
            corr_rows.append({"metric": "back-translation chrF (no reference)", "subset": name, "n": len(sub),
                              "spearman": round(r_s[0], 3), "spearman_p": r_s[1], "pearson": round(r_p[0], 3)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv("results/rating_vs_chrf_correlation.csv", index=False)
    display(corr)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(ev))
    axes[0].scatter(fl["ref_chrf"], fl["rating"] + jit[:len(fl)], s=16, color="#2a6f97")
    axes[0].set_xlabel("chrF vs reference"); axes[0].set_ylabel("LLM-judge rating (1-5)"); axes[0].set_title("FLORES rated", fontsize=10)
    axes[1].scatter(ev["bt_chrf"], ev["rating"] + jit, s=16, c=np.where(ev["src_type"] == "flores", "#2a6f97", "#e07a5f"))
    axes[1].set_xlabel("back-translation chrF"); axes[1].set_title("all rated (orange = own set)", fontsize=10)
    for a in axes: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("results/rating_vs_chrf.png", dpi=150); plt.show()
""")

code("""
def flag(df, tau):
    return (df["hard"].fillna("") != "") | (df["bt_chrf"] < tau)

def scores(df, flags, bad):
    tp = int((flags & bad).sum()); fp = int((flags & ~bad).sum())
    fn = int((~flags & bad).sum()); tn = int((~flags & ~bad).sum())
    return {"n": len(df), "bad": int(bad.sum()), "caught_bad": tp, "missed_bad": fn,
            "false_alarms": fp, "good_not_flagged": tn,
            "recall_bad": round(tp / max(tp + fn, 1), 3),
            "false_alarm_rate": round(fp / max(fp + tn, 1), 3),
            "precision": round(tp / max(tp + fp, 1), 3) if tp + fp else None}

def f1(df, tau, bad):
    fl_ = flag(df, tau); tp = (fl_ & bad).sum(); fp = (fl_ & ~bad).sum(); fn = (~fl_ & bad).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

if ratings_filled():
    perm = np.random.default_rng(2026).permutation(len(ev))
    calib, test = ev.iloc[perm[: len(ev) // 2]], ev.iloc[perm[len(ev) // 2:]]
    out_rows, thresholds = [], {}
    for cut in (3, 2):
        bad_c, bad_t = calib["rating"] <= cut, test["rating"] <= cut
        grid = sorted(set(np.round(calib["bt_chrf"], 1)) | {0.0, 101.0})
        best = max(grid, key=lambda t: (f1(calib, t, bad_c), -t))  # ties -> lower threshold (fewer alarms)
        thresholds[f"bad_if_rating_le_{cut}"] = float(best)
        for system, flags in [("always trust (baseline)", pd.Series(False, index=test.index)),
                              ("hard checks only", flag(test, -1)),
                              (f"Senim: hard checks + back-translation < {best:.1f}", flag(test, best))]:
            out_rows.append({"bad_definition": f"rating <= {cut}", "split": "test", "system": system,
                             **scores(test, flags, bad_t)})
    trust_eval = pd.DataFrame(out_rows)
    trust_eval.to_csv("results/trust_check_eval.csv", index=False)
    json.dump({"bt_chrf_threshold": thresholds["bad_if_rating_le_3"], "chosen_on": "calibration half of data/ratings.csv",
               "bad_definition": "rating_1_5 <= 3", "all": thresholds, "n_calibration": len(calib), "n_test": len(test)},
              open("results/threshold.json", "w"), indent=2)
    print("thresholds chosen on calibration:", thresholds)
    display(trust_eval)
""")

md("## 9. Text summary of all results")

code("""
import glob
for f in ["results/chrf_by_language.csv", "results/worst5_kaz.csv", "results/rating_vs_chrf_correlation.csv",
          "results/trust_check_eval.csv", "results/threshold.json"]:
    if os.path.exists(f):
        print(f"===== {f}"); print(open(f, encoding="utf-8").read())
print(f"Spearman(back-translation chrF, reference chrF), FLORES kaz n={len(flores)}: {rho:.3f} (p={p:.1e})")
print("FLORES sentences flagged by hard checks:", int((flores["hard_check_reasons"] != "").sum()), "of", len(flores))
if len(own):
    print("own sentences flagged by hard checks:", int((own["hard_check_reasons"] != "").sum()), "of", len(own))
    print("own set mean back-translation chrF:", round(own["bt_chrf"].mean(), 1), "| FLORES:", round(flores["bt_chrf"].mean(), 1))
""")

md("## 10. Save everything")

code("""
import shutil
print(sorted(os.listdir("results")))
if IN_COLAB:
    shutil.make_archive("/content/senim_results", "zip", "results")
    print("zip: /content/senim_results.zip")
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
Path(__file__).resolve().parent.parent.joinpath("notebook.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("notebook.ipynb written with", len(CELLS), "cells")
