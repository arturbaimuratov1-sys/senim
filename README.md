# Senim · Сенім — a translator that tells you when not to trust it

**Veritas AI International High School Hackathon 2026 · Track: AI + Languages · Brief 4.4.2 "Translation quality test"**

> **This project is a prototype and is not medical advice.**

English → Kazakh machine translation, plus a warning label that says when a translation should **not** be trusted, and why.

- Notebook (runs on a free Colab T4 GPU): [`notebook.ipynb`](notebook.ipynb) · [open in Colab](https://colab.research.google.com/github/arturbaimuratov1-sys/senim/blob/main/notebook.ipynb)
- Demo app: [`app.py`](app.py) (Gradio)
- Trust-check code: [`senim.py`](senim.py) · tests: [`tests/test_hard_checks.py`](tests/test_hard_checks.py)
- All result tables: [`results/`](results)

---

## The problem

A grandmother in Kazakhstan gets a medicine leaflet in English, or a parent gets a letter from an international school. They don't read English, so they paste the text into a free translator. The translation *looks* fluent. But if it turned "every 8 hours" into "every 6 hours", or "if you notice bleeding" into "if you do **not** have bleeding", they have no way to know.

Translators always give an answer and never say "I'm not sure". For Kazakh, a lower-resource language, mistakes are more common than for French or Russian. Our question: **can we warn the reader when a Kazakh translation is risky, using only free tools and no Kazakh reference translation?**

## What we built

1. **A translation quality test** (the brief): 150 sentences from the FLORES+ benchmark, translated by Meta's open NLLB-200 (600M) model into Kazakh, Russian and French, scored with chrF against professional human translations, and compared with a "do nothing" baseline (just show the English).
2. **Our own test set**: 50 English sentences of the kind our users actually meet: 25 from medicine leaflets, 25 from school letters, many with doses, times and dates.
3. **Senim's trust check**: for each translation, a label **"likely OK"** or **"do not rely on this"**, with the reasons in Kazakh and English.
4. **A demo app** where you paste English text and get the Kazakh translation, the trust label, and a back-translation to check against.

## What we added (beyond the brief)

- **A safety check for the things that hurt most**: numbers, dosage units (mg, ml, °C…), clock times, months and weekdays must survive translation. Number words count too ("two" = "екі" = 2), and "6:00 PM" is allowed to become "18:00".
- **A no-reference quality signal**: translate the Kazakh back into English and compare it with the original (back-translation chrF). This works on any new text, where no human reference exists.
- **An honest evaluation of the trust check**: the warning threshold is tuned on one half of the rated translations and tested on the other half, against the baseline "always trust".

## How it works

```
English text ──► NLLB-200 ──► Kazakh text ──► NLLB-200 ──► back to English
                                   │                              │
                                   ▼                              ▼
                      hard checks vs. original          chrF(back-translation, original)
             (numbers, units, times, dates)               below threshold 55.9?
                                   └────────────┬─────────────────┘
                                                ▼
                       any problem → ⚠️ "Do not rely on this translation" + reasons
                       none        → ✅ "Likely OK"
```

- **chrF** compares character n-grams between a translation and a reference, on a 0–100 scale. We use sacreBLEU's implementation.
- **The threshold** (55.9) was chosen on the *calibration* half of the rated translations only, to maximise F1 for catching bad translations. The *test* half was never used to choose anything.
- **The hard-check rules** were revised once, after the first Colab run and before any rating existed. The fixes came from looking at flagged *translations* (not ratings), including some sentences that later ended up in the rated set (for example "6:00 PM" → "18:00" and the colloquial weekday "шейшембі"). So the hard-check results on those sentences are slightly optimistic. Other fixes: the first version wrongly treated "first aid" as the number 1 and missed that Kazakh "қыркүйек" (September) becomes "қыркүйегінде". The rules were frozen before rating.

## Results

All numbers come from the Colab run in `notebook.ipynb` (NLLB-200 distilled 600M, beam 4, T4 GPU; translating 150 sentences took 12–14 seconds per language).

### 1. Translation quality on FLORES+ (n = 150, corpus chrF)

| Language | NLLB-200 600M | Baseline: leave English |
|---|---|---|
| Kazakh | **52.1** | 1.4 |
| Russian | **56.0** | 1.0 |
| French | **69.2** | 25.5 |

Kazakh scores lowest of the three: the model is clearly weaker on Kazakh than on French. Translating is still far better than showing English (chrF 1.4), but 52 is not "safe to trust blindly".

**The 5 worst Kazakh sentences** ([`results/worst5_kaz.csv`](results/worst5_kaz.csv), sentence chrF 16.7–24.5) show typical failure types:
- iPods became "iPhone" (Айфон);
- a 21–20 rugby result ending a winning streak came out as the team "won 15 games";
- in one case the whole second sentence was dropped.

### 2. Who rated the translations

We rated 90 Kazakh translations (all 50 of ours and 40 random FLORES ones) on a 1–5 scale:
- 5 = correct;
- 3 = meaning partly wrong or something missing;
- 1 = useless or dangerous.

**The ratings were written down by Claude Opus 5.5 acting as an LLM judge** (column `rater` in `data/ratings.csv`). The project author then went through all 90 translations aloud, judged them independently without looking at the LLM's numbers, and reports that their judgement agreed with the LLM's ratings and error types. The author's check was not written down, so there is no file with separate human scores and we cannot report an agreement statistic. Read every number below as *agreement with an LLM judge that the author found consistent with their own reading*, not as an independent human rating. The author's overall impression of the translations: "the meaning mostly comes through, but not exactly, and some Russian words appear".

| | n | mean rating | bad (≤ 3) | serious (≤ 2) |
|---|---|---|---|---|
| Our medicine sentences | 25 | – | 9 | 4 |
| Our school sentences | 25 | – | 7 | 3 |
| Our set, total | 50 | 3.92 | 16 | 7 |
| FLORES | 40 | 3.67 | 16 | 8 |
| **All** | **90** | **3.81** | **32** | **15** |

Main error types among the 32 bad translations:

| Error type | Count | Example |
|---|---|---|
| wrong word | 21 | "consult a doctor" → "give advice to the doctor" |
| dropped content | 7 | |
| flipped negation | 2 | "if you notice unusual bleeding" → "if you do **not** have usual bleeding" |
| wrong date | 1 | Friday → Sunday |
| grammar | 1 | |

These are exactly the errors a reader can't spot.

### 3. Does chrF agree with the ratings? (Spearman correlation)

| Signal | Subset | n | Spearman ρ | p |
|---|---|---|---|---|
| chrF vs Kazakh reference | FLORES rated | 40 | 0.45 | 0.003 |
| back-translation chrF (no reference needed) | all rated | 90 | 0.45 | < 0.001 |
| back-translation chrF | our set | 50 | 0.42 | 0.002 |
| back-translation chrF | FLORES rated | 40 | 0.47 | 0.002 |

Both are moderate signals, far from perfect. On all 150 FLORES sentences, back-translation chrF and reference chrF agree only weakly: ρ = 0.30 (p = 0.0002).

### 4. Does the trust check catch bad translations? (test half only, n = 45, 16 bad)

| System | Bad caught | Missed | False alarms (of 29 good) | Precision |
|---|---|---|---|---|
| Always trust (baseline) | 0 / 16 | 16 | 0 | – |
| Hard checks only | 1 / 16 | 15 | 0 | 100% |
| **Senim: hard checks + back-translation < 55.9** | **12 / 16 (75%)** | 4 | 11 (38%) | 52% |

Senim caught 12 of the 16 bad translations that "always trust" would have shown as fine. The price is many false alarms: 11 of 29 good translations were also flagged. On a test set this small, the uncertainty is large: 95% Wilson interval 51–90% for the catch rate and 23–56% for the false-alarm rate.

**Weak spots, shown as they are:**
- **Hard checks catch little.** On our sentences the model kept numbers, times and dates right almost every time: hard checks flagged 1 of 50, the Friday → Sunday error. The real damage was wrong words and flipped negation, which only back-translation sometimes notices.
- **Serious errors only (rating ≤ 2) are poorly caught.** With a separate threshold tuned for this stricter definition (31.4), Senim caught 1 of 7 on the test half. Back-translation hides some errors: when NLLB makes a mistake, translating back often "repairs" it.
- **The labels were written by an LLM; the human check is informal.** The author says they checked all 90 aloud and agreed, but nothing was recorded, so this is not a measured human rating. The author is also not necessarily a fluent Kazakh rater, and the same AI that wrote the labels helped design the check, which can bias the evaluation.
- **Small data.** 90 rated sentences, 45 in the test half.

## What we'd do next

1. **Recorded human ratings**: 2–3 fluent Kazakh speakers write down their own 1–5 scores for the same 90 translations (a blank sheet is ready: `data/ratings_human.xlsx`), and we report how often they agree with each other and with the LLM judge.
2. **A negation check**: compare "not / do not / never" in the source with Kazakh negative forms (-ма/-ме, емес, жоқ). This was the most dangerous error and our checks missed it.
3. **A bigger model** (NLLB-200 1.3B/3.3B) or several translators voting: if two systems disagree, warn.
4. **A "Russian loanword" detector**, since the author noticed Russian words in the output.
5. **Test with real users**: is a warning useful, or does a 38% false-alarm rate teach people to ignore it?

## Run it yourself

- **Notebook:** open it [in Colab](https://colab.research.google.com/github/arturbaimuratov1-sys/senim/blob/main/notebook.ipynb) and choose Runtime → Change runtime type → **T4 GPU**. Add your Hugging Face token as the Colab secret `HF_TOKEN`; FLORES+ is gated, so accept its terms on Hugging Face first. Then Runtime → Run all.
- **App:** `pip install -r requirements.txt && python app.py`, or `!python app.py` in Colab for a public link.
- **Tests:** `pip install pytest && python -m pytest tests/`

## Sources

Full list with links and licences: [SOURCES.md](SOURCES.md).

- **Data:** FLORES+ (Open Language Data Initiative, CC BY-SA 4.0); own test set written by the author.
- **Model:** NLLB-200 distilled 600M (Meta, CC BY-NC 4.0).
- **Metric:** chrF (Popović 2015) via sacreBLEU.
- **Libraries:** PyTorch, Transformers, Datasets, Gradio, pandas, SciPy, Matplotlib. Platform: Google Colab.
- **AI tools:** Claude Code (Claude Opus 5.5) wrote most of the code, ran the notebook in Colab, gave the LLM-judge ratings, and drafted this write-up.

*This project is a prototype and is not medical advice.*
