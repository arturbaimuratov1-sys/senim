# Senim · Сенім — a translator that tells you when not to trust it

**Veritas AI International High School Hackathon 2026 · Track: AI + Languages · Brief 4.4.2 "Translation quality test"**

> **This project is a prototype and is not medical advice.**

English → Kazakh machine translation, plus a warning label that says when a translation should **not** be trusted, and why.

- Notebook (runs on a free Colab T4 GPU): [`notebook.ipynb`](notebook.ipynb) · [open in Colab](https://colab.research.google.com/github/arturbaimuratov1-sys/senim/blob/main/notebook.ipynb)
- Demo app: [`app.py`](app.py) (Gradio)
- Trust-check code: [`senim.py`](senim.py) · tests: [`tests/test_hard_checks.py`](tests/test_hard_checks.py)
- All result tables and charts: [`results/`](results). Full translations: [`results/flores_translations.csv`](results/flores_translations.csv), [`results/own_set_translations.csv`](results/own_set_translations.csv)
- Ratings side by side (human and LLM judge), easy to read in Excel: [`results/ratings_view.xlsx`](results/ratings_view.xlsx)

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
             (numbers, units, times, dates)               below threshold 63.0?
                                   └────────────┬─────────────────┘
                                                ▼
                       any problem → ⚠️ "Do not rely on this translation" + reasons
                       none        → ✅ "Likely OK"
```

- **chrF** compares character n-grams between a translation and a reference, on a 0–100 scale. We use sacreBLEU's implementation.
- **The threshold** (63.0) was chosen on the *calibration* half of the human-rated translations only, to maximise F1 for catching bad translations. The *test* half was never used to choose anything.
- **The hard-check rules** were revised once, after the first Colab run and before any rating existed. The fixes came from looking at flagged *translations* (not ratings), including some sentences that later ended up in the rated set (for example "6:00 PM" → "18:00" and the colloquial weekday "шейшембі"). So the hard-check results on those sentences are slightly optimistic. Other fixes: the first version wrongly treated "first aid" as the number 1 and missed that Kazakh "қыркүйек" (September) becomes "қыркүйегінде". The rules were frozen before rating.

## Results

All numbers come from the Colab run in `notebook.ipynb` (NLLB-200 distilled 600M, beam 4, T4 GPU; translating 150 sentences took 12–14 seconds per language).

### 1. Translation quality on FLORES+ (n = 150, corpus chrF)

| Language | NLLB-200 600M | Baseline: leave English |
|---|---|---|
| Kazakh | **52.1** | 1.4 |
| Russian | **56.0** | 1.0 |
| French | **69.2** | 25.5 |

![chrF by language](results/chrf_by_language.png)

Kazakh scores lowest of the three: the model is clearly weaker on Kazakh than on French. Translating is still far better than showing English (chrF 1.4), but 52 is not "safe to trust blindly".

**The 5 worst Kazakh sentences** ([`results/worst5_kaz.csv`](results/worst5_kaz.csv), sentence chrF 16.7–24.5) show typical failure types:
- iPods became "iPhone" (Айфон);
- a 21–20 rugby result ending a winning streak came out as the team "won 15 games";
- in one case the whole second sentence was dropped.

### 2. Human ratings of 90 translations

We rated 90 Kazakh translations (all 50 of ours and 40 random FLORES ones) on a 1–5 scale:
- 5 = correct;
- 3 = meaning partly wrong or something missing;
- 1 = useless or dangerous.

**Who rated.** The project author rated all 90 in Excel and, for every translation, wrote what is wrong and how it should read (`data/ratings_human_filled.xlsx`, also merged into `data/ratings.csv`). The author says they did this without looking at the AI's numbers. Separately and earlier, Claude Opus 5.5 rated the same 90 as an LLM judge; those scores are kept in the `llm_*` columns for comparison. There is only one human rater, so we cannot measure agreement between people.

| | n | mean (human) | bad (≤ 3) | serious (≤ 2) |
|---|---|---|---|---|
| Our medicine sentences | 25 | 3.40 | 13 | 5 |
| Our school sentences | 25 | 3.60 | 8 | 3 |
| FLORES | 40 | 3.15 | 23 | 11 |
| **All** | **90** | **3.34** | **44** | **19** |

**Medicine leaflets came out worse than school letters**: 13 of 25 bad, against 8 of 25. This is bad news for the people we care about most.

What went wrong in the 44 bad translations (from the author's comments):

| Problem | Count | Example |
|---|---|---|
| meaning distorted | 19 | "consult a doctor" → "give advice to the doctor"; developed countries → "developing" |
| wrong or unnatural word | 12 | red cabbage → "red vegetable"; exam → "class" |
| content dropped | 8 | the translation stops halfway: "do not take with alcohol because it…" |
| grammar or logic | 3 | "so that" → "therefore" |
| wrong date | 1 | Friday → Sunday |
| name or term | 1 | |

Two of the "meaning distorted" cases are **flipped negations**, the most dangerous kind:
- "if you notice unusual bleeding" → "if you do **not** have usual bleeding";
- "your child may miss the trip" → "may **not** miss the trip".

### 3. Human vs. LLM judge

| | |
|---|---|
| Same score | 47 / 90 |
| Within 1 point | 89 / 90 |
| Spearman ρ | 0.87 |
| Quadratic-weighted κ | 0.80 |
| Agree on "bad (≤ 3)" | 76 / 90 (κ = 0.69) |
| Agree on "serious (≤ 2)" | 86 / 90 (κ = 0.86) |
| Mean score: human / LLM | 3.34 / 3.81 |

The two raters order the translations very similarly, but **the human is stricter**: lower in 42 cases, higher in 1. The LLM judge called 32 translations bad, the human 44. All but one of the LLM's "bad" were also bad for the human. In short, an LLM judge is a usable first filter, but it is too lenient to replace a human. Full confusion table: [`results/human_vs_llm_confusion.csv`](results/human_vs_llm_confusion.csv).

### 4. Does chrF agree with human ratings? (Spearman correlation)

| Signal | Subset | n | Spearman ρ | p |
|---|---|---|---|---|
| chrF vs Kazakh reference | FLORES rated | 40 | 0.48 | 0.002 |
| back-translation chrF (no reference needed) | all rated | 90 | 0.43 | < 0.001 |
| back-translation chrF | our set | 50 | 0.35 | 0.012 |
| back-translation chrF | FLORES rated | 40 | 0.51 | < 0.001 |

![Human rating vs chrF](results/rating_vs_chrf.png)

Both are moderate signals, far from perfect, and back-translation is weakest on our own medicine and school sentences (0.35). In the right-hand plot, one of our sentences got a perfect back-translation score of 100 but a human rating of 2: the back-translation "repaired" the error, so the check could not see it. It is `own_19`: "consult a doctor" became "give advice to the doctor", and translating back gave exactly the original English. On all 150 FLORES sentences, back-translation chrF and reference chrF agree only weakly: ρ = 0.30 (p = 0.0002).

![Back-translation chrF vs reference chrF](results/bt_vs_reference_chrf.png)

### 5. Does the trust check catch bad translations? (human labels, test half only, n = 45)

The threshold was chosen on the other 45 sentences.

**"Bad" = rating ≤ 3** (25 of the 45 test sentences are bad):

| System | Bad caught | False alarms (of 20 good) | Precision |
|---|---|---|---|
| Always trust (baseline) | 0 / 25 | 0 | – |
| Always warn (for comparison) | 25 / 25 | 20 (100%) | 56% |
| Hard checks only | 1 / 25 | 0 | 100% |
| **Senim (threshold 63.0)** | **19 / 25 (76%)** | 13 (65%) | 59% |

**"Bad" = rating ≤ 2, serious errors only** (8 of 45):

| System | Bad caught | False alarms (of 37 good) | Precision |
|---|---|---|---|
| Always trust (baseline) | 0 / 8 | 0 | – |
| Hard checks only | 1 / 8 | 0 | 100% |
| **Senim (threshold 47.0)** | **6 / 8 (75%)** | 9 (24%) | 40% |

**How to read this honestly:**
- **Serious errors: the check helps.** It catches 6 of 8 while warning on only 24% of acceptable translations. When it warns, the translation is seriously wrong 40% of the time, against 18% (8 of 45) if you picked at random.
- **The broad "≤ 3" definition: the check barely beats warning on everything.** By the human's strict standard more than half of the translations are bad, so a warning is right only 59% of the time, against 56% for "always warn". It catches 76%, but also flags 65% of good translations.
- **The numbers are uncertain.** With 45 test sentences, the 95% Wilson interval for "6 of 8 caught" is 41–93%.
- **Compared with the LLM labels:** with the LLM judge's ratings, the same procedure caught 12 / 16 bad (≤ 3) with 38% false alarms, but only 1 / 7 serious ones ([`results/trust_check_eval_llm.csv`](results/trust_check_eval_llm.csv)). Who rates changes the conclusion, which is one more reason human labels matter.

**Weak spots, shown as they are:**
- **Hard checks catch little.** NLLB kept numbers, times and dates right almost every time: hard checks flagged 1 of our 50 sentences, the Friday → Sunday error. The real damage was distorted meaning and wrong words, which only back-translation sometimes notices.
- **Back-translation can hide errors.** When NLLB makes a mistake, translating back with the same model often "repairs" it.
- **One human rater** who is the project author, not an independent professional, and who saw the translations while building the project. The procedure was first run with the LLM labels and then repeated unchanged with the human labels.
- **Small data.** 90 rated sentences, 45 in the test half.

## What we'd do next

1. **More human raters**: 2–3 more fluent Kazakh speakers rate the same 90 translations (blank sheet: `data/ratings_human.xlsx`), so we can measure agreement between people, not just between one person and an LLM.
2. **A negation check**: compare "not / do not / never" in the source with Kazakh negative forms (-ма/-ме, емес, жоқ). This was the most dangerous error and our checks missed it.
3. **A bigger model** (NLLB-200 1.3B/3.3B) or several translators voting: if two systems disagree, warn.
4. **A "Russian loanword" detector**, since the author noticed Russian words in the output.
5. **Test with real users**: is a warning useful, or does a high false-alarm rate teach people to ignore it?
6. **Use the author's corrections**: every bad translation now has a human-written correct version (`human_correction`), which can serve as a small Kazakh reference set for medicine and school texts.

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
- **Ratings:** the author (human, primary); Claude Opus 5.5 as an LLM judge (secondary, for comparison).
- **AI tools:** Claude Code (Claude Opus 5.5) wrote most of the code, ran the notebook in Colab, gave the LLM-judge ratings, and drafted this write-up.

*This project is a prototype and is not medical advice.*
