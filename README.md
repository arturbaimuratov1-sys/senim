# Senim · Сенім — a translator that tells you when not to trust it

**Track:** Languages · **Team:** see the submission form

> **This project is a prototype and is not medical advice.**

**Main takeaway:** a free translator got **44 of 90** Kazakh translations partly or fully wrong (rated by a Kazakh speaker), and the errors read fluently. Senim warns about **6 of 8 serious errors** (baseline "always trust": 0), and choosing the best of 8 candidate translations improves the translation itself by **+2.6 chrF** on held-out test data.

[Notebook with outputs](notebook.ipynb) · [Full report](REPORT.md) · [Sources](SOURCES.md) · [Demo app](app.py)

## The problem
A grandmother in Kazakhstan gets a medicine leaflet in English; a parent gets a letter from school. They paste it into a free translator and get fluent Kazakh, with no way to see that a sentence now says the opposite.

## What we built
English text in → Kazakh translation out, with a label **✅ likely OK** or **⚠️ do not rely on this**, the reason in Kazakh, a back-translation to check against and, when it warns, an improved translation.

## What we added
Started from suggested project **4.4.2 (Translation quality test)**. Added: **our own user** (parents and grandparents who don't read English); **our own data** (50 medicine and school sentences written by the author, and 90 translations rated 1–5 by the author, a Kazakh speaker, each with a corrected version); **our own question** (can people safely rely on the translation?), answered with a trust check and a way to reduce the errors it targets.

## How it works
- **Data:** FLORES+ (professional translations): 150 devtest sentences for testing, 997 dev sentences for tuning; plus our 50 sentences.
- **Model:** NLLB-200 distilled 600M (free, open).
- **Trust check:** numbers, units, times, dates and negations must match the original, and the Kazakh translated back to English must stay close to the source (chrF). The threshold was chosen on half of the ratings and tested on the other half.
- **Improvement:** generate 8 candidates (NLLB 600M and 1.3B) and keep the one that passes the checks and back-translates best. Rules were fixed in advance and tuned on FLORES dev only.

## Results
**Translation quality** (FLORES+ devtest, n = 150, chrF):

| | Kazakh | Russian | French |
|---|---|---|---|
| Baseline: leave English | 1.4 | 1.0 | 25.5 |
| NLLB 600M | 52.1 | 56.0 | 69.2 |
| **Senim, best of 8** | **54.8** (+2.6, 95% CI 1.2–4.0) | — | — |

**Trust check vs. the author's ratings** (held-out half, n = 45; threshold tuned per row, the app uses 63):

| Errors | Always trust | Senim |
|---|---|---|
| Serious (rating ≤ 2), 8 cases | 0 caught | **6 caught**, 24% false alarms |
| Any (rating ≤ 3), 25 cases | 0 caught | 19 caught, 65% false alarms (barely better than "always warn") |

**Real examples** (model output next to the author's correction):

| English | NLLB says | Should say |
|---|---|---|
| …if you notice **unusual** bleeding | "if you do **not** have normal bleeding" | әдеттен тыс қан кетуді байқасаңыз |
| …your child **may miss** the trip | "may **not** miss the trip" | сапарға бара алмауы мүмкін |
| …for **Friday's** outdoor activity | "on **Sunday**" (жексенбі) | жұмадағы |

**Where it fails:** medicine sentences fail more often than school ones (13 of 25 vs 8 of 25 bad). Back-translation can "repair" an error and hide it ("consult a doctor" → "give advice to the doctor" back-translated perfectly). Best-of-8 is about 4× slower and was not re-rated by a human. One rater; 45 sentences in the held-out half. Faster variants, a code review and a failed combined model are in [REPORT §6–8](REPORT.md).

**Informal feedback** (reported by the author, not a study): a pharmacist who also reads English recommended the project and felt the AI translated correctly in about 9 of 10 cases. That is a general impression and more optimistic than our ratings, and we don't know which translations were seen. An elderly woman who reads neither English nor French confirmed the problem and said she doesn't trust general chatbots for medicine.

## What we'd do next
1. Ask the pharmacist and more Kazakh speakers to rate the same 90 translations, to measure agreement between people.
2. Re-rate the best-of-8 output and calibrate the warning on it.

## Sources
Full list with links: [SOURCES.md](SOURCES.md). FLORES+ (CC BY-SA 4.0), NLLB-200 600M and 1.3B (Meta, CC BY-NC 4.0), sacreBLEU, Hugging Face Transformers, Gradio, Google Colab, Kaggle. **AI tools:** Claude Code (Claude Opus 5.5) wrote most of the code, ran the notebooks and drafted this write-up.
