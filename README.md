# Senim · Сенім — a translator that tells you when not to trust it

**Track:** Languages · **Team:** see the submission form

> **This project is a prototype and is not medical advice.**

**Main takeaway:** a free translator got **44 of 90** Kazakh translations wrong by a Kazakh speaker's rating, and the errors read fluently. Senim warns about **6 of 8 serious errors** (baseline "always trust": 0), and picking the best of 8 candidate translations makes the translation itself better (**+2.6 chrF** on held-out test data).

[Notebook with outputs](notebook.ipynb) · [Full report with every number](REPORT.md) · [Sources](SOURCES.md) · [Demo app](app.py)

## The problem
A grandmother in Kazakhstan gets a medicine leaflet in English; a parent gets a letter from school. They paste it into a free translator and get fluent Kazakh, with no way to see that a sentence now says the opposite.

## What we built
English text in → Kazakh translation out, plus a label **✅ likely OK** or **⚠️ do not rely on this**, with the reason in Kazakh, and a back-translation to check against. When it warns, it also offers an improved translation.

## What we added
Started from suggested project **4.4.2 (Translation quality test)**. Added:
- **Own user:** parents and grandparents who don't read English.
- **Own data:** 50 medicine and school sentences written by the author. 90 translations rated 1–5 by the author (a Kazakh speaker), each with a corrected version.
- **Own question:** can people safely rely on the translation? We built a trust check, and then a way to **reduce** the errors it targets (numbers, dates, negation).

## How it works
- **Data:** FLORES+ (English with professional Kazakh, Russian and French translations): 150 devtest sentences for testing, 997 dev sentences for tuning; plus our 50 sentences.
- **Model:** NLLB-200 distilled 600M, free and open.
- **Trust check:** numbers, units, times, dates and negations must match the original. The Kazakh is also translated back to English and compared with the original (chrF). The threshold was chosen on half of the ratings and tested on the other half.
- **Improvement (v3):** generate 8 candidates (4 from NLLB 600M, 4 from NLLB 1.3B) and keep the one that passes the checks and translates back best. Rules were fixed in advance and tuned on FLORES dev only.

## Results
**Translation quality** (FLORES+ devtest, n = 150, chrF):

| | Kazakh | Russian | French |
|---|---|---|---|
| Baseline: leave English | 1.4 | 1.0 | 25.5 |
| NLLB 600M | 52.1 | 56.0 | 69.2 |
| **Senim v3 (best of 8)** | **54.8** (+2.6, 95% CI 1.2–4.0) | — | — |

**Trust check vs. human ratings** (held-out half, n = 45):

| Errors | Always trust | Senim |
|---|---|---|
| Serious (rating ≤ 2), 8 cases | 0 caught | **6 caught**, 24% false alarms |
| Any (rating ≤ 3), 25 cases | 0 caught | 19 caught, but 65% false alarms (barely better than "always warn") |

The app also runs a negation check: with it, 7 of 8 serious errors are caught, at 38% false alarms.

**Real examples** (model output next to the author's correction):

| English | NLLB says | Should say |
|---|---|---|
| …call your doctor if you notice **unusual** bleeding | "if you do **not** have normal bleeding" | әдеттен тыс қан кетуді байқасаңыз |
| …your child **may miss** the trip | "may **not** miss the trip" | сапарға бара алмауы мүмкін |
| …for **Friday's** outdoor activity | "on **Sunday**" (жексенбі) | жұмадағы |

**Faster and checked (v4, [REPORT §8](REPORT.md)):**
- An AI code review found 5 bugs in the checks (e.g. "6 pm" vs "сағат 18-де"). All are fixed with tests, and none changes any reported result.
- A CTranslate2 backend is 3.2× faster on CPU at the same chrF. It rewords 2 in 3 outputs, which can shift the calibrated warning, so it is opt-in.
- Running best-of-8 only for risky sentences saved 30% of the time but missed our pre-set bar.

**Where it fails:**
- Medicine sentences fail more often than school ones (13 of 25 vs 8 of 25 bad).
- Back-translation sometimes "repairs" an error, so the check misses it: "consult a doctor" → "give advice to the doctor" back-translated perfectly.
- Best-of-8 is 4× slower, and it was not re-rated by a human.
- One human rater; 45 sentences in the held-out half.

## What real people said (informal feedback, not a study)
Reported by the author, who asked two people:
- **A pharmacist who also reads English** reviewed the project and gave a positive recommendation. In the pharmacist's opinion the AI translated correctly in about 9 of 10 cases. This is a general impression, not a sentence-by-sentence rating, and it is more optimistic than our own rating (44 of 90 translations judged wrong by the author, who counted any change of meaning; 13 of 25 medicine sentences). We do not know which translations the pharmacist looked at.
- **An elderly woman who reads neither English nor French** confirmed that the problem is real. She said she does not fully trust general chatbots such as ChatGPT for medicine, because they are not specialised in pharmacy or translation. A warning label that says when not to trust a translation answers exactly this doubt.

## What we'd do next
1. Ask the pharmacist and more Kazakh speakers to rate the same 90 translations sentence by sentence, to measure agreement between people.
2. Re-rate the best-of-8 output and calibrate the warning on it.

## Sources
Full list with links: [SOURCES.md](SOURCES.md).
- **Data and models:** FLORES+ (CC BY-SA 4.0); NLLB-200 600M and 1.3B (Meta, CC BY-NC 4.0); LaBSE.
- **Tools:** sacreBLEU, Hugging Face Transformers, Gradio, Google Colab, Kaggle.
- **AI tools:** Claude Code (Claude Opus 5.5) wrote most of the code, ran the notebooks and drafted this write-up.
