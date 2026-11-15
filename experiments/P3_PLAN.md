# Senim v3 — pre-registered experiment plan

Written **before** running anything. Results go to `results/v3_*`.

## Question
Can Senim *reduce* translation errors instead of only warning about them?
(Brief 4.4.2 idea: "find one type of error … and test a way to reduce it".)

## Systems (English → Kazakh)
| id | system |
|---|---|
| A | NLLB-200 600M, beam 4 (the current Senim translator, the one the human rated) |
| S | A, but translated sentence by sentence (`translate_split`) |
| B | NLLB-200 1.3B, beam 4 |
| R600 | **select_best** over the 4 best beams of 600M |
| R | **select_best** over the 4 best beams of 600M + the 4 best beams of 1.3B |

`select_best`: drop candidates that fail the hard checks (numbers, units, times, dates) or the negation check,
then keep the one whose back-translation (600M, Kazakh → English) has the highest chrF with the English source.

## Data
1. **Development:** FLORES+ `dev` split, all 997 English–Kazakh pairs. Never used before in this project.
   All decisions are made here.
2. **Final, used once:** the same 150 FLORES+ `devtest` sentences as in `notebook.ipynb`.
3. **Domain check:** the 90 human-rated sentences, scored against the author's corrected translations
   (`human_correction`). Caveat: the corrections were made by editing system A's output, so this test favours A.

## Metrics
- corpus chrF against the Kazakh reference;
- mean sentence-chrF difference vs A with a 95% paired-bootstrap interval (1000 resamples);
- number of translations failing the hard checks / the negation check;
- translation time.

## Decision rule (fixed now)
A system replaces A in the app only if on **dev** its mean sentence-chrF gain over A has a 95% interval above 0
**and** it does not increase hard-check or negation failures. Among qualifying systems, the one with the largest gain wins;
if gains are within 0.3 chrF, the faster one wins. The winner is then run **once** on devtest-150 and that number is reported.
If nothing qualifies, Senim keeps A and the README says so in one line.
