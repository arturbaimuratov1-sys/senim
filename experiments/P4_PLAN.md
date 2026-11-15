# Senim v4 — pre-registered plan: faster without losing quality

Written **before** running. Results go to `results/v4_*`.

## Question
v3 (best of 8 candidates, "R") improves Kazakh translation by +2.15 chrF on FLORES+ dev, but is ~4.4× slower than the plain model.
Can we keep most of the gain while paying the extra cost only for risky sentences?

## Systems (English → Kazakh)
| id | system |
|---|---|
| A | NLLB 600M, beam 4 |
| R | v3: best of 8 (4 beams of 600M + 4 beams of 1.3B), for every sentence |
| C | **cascade** (`smart_translate`): A for sentences that pass the app's trust check; R only for the rest |

The trust check used by C is the app's, unchanged: no number/unit/time/date problem, no negation mismatch,
back-translation chrF ≥ **63.0** (the threshold already chosen on the human ratings; not tuned here).

## Data
FLORES+ dev, all 997 pairs (decisions); then the winner and A **once** on the notebook's 150 devtest sentences.

## Timing
Both models are loaded and warmed up before timing. Each system is timed from scratch (no shared caches between systems),
on the same Kaggle T4 GPU.

## Metrics
corpus chrF; mean sentence-chrF gain vs A with a 95% paired-bootstrap interval; seconds per sentence; share of sentences
sent to stage 2 (C only).

## Decision rule (fixed now)
C replaces R in the app if its gain over A has a 95% interval above 0 **and** it needs at most 60% of R's time.
Otherwise the app keeps R (and the README says so in one line).
