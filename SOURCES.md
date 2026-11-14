# Sources

Everything this project uses, with links. Licences are those stated by each source.

## Data

| What | Where | Licence / note |
|---|---|---|
| FLORES+ (devtest split; `eng_Latn`, `kaz_Cyrl`, `rus_Cyrl`, `fra_Latn`) | https://huggingface.co/datasets/openlanguagedata/flores_plus | CC BY-SA 4.0, gated (terms accepted on Hugging Face). Maintained by the Open Language Data Initiative, https://oldi.org |
| Original FLORES-200 benchmark | NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*, https://arxiv.org/abs/2207.04672 | — |
| `data/own_set.csv`: 50 English sentences (25 medicine, 25 school) | Written by the project author. Claude Code suggested a list of topics; the sentences themselves are the author's. | — |
| `data/ratings.csv`: 1–5 quality ratings of 90 Kazakh translations | Given by **Claude Opus 5.5 acting as an LLM judge** (column `rater`), not by a human rater. Script: `tools/fill_llm_ratings.py`. | See README, "Limitations" |

## Model

| What | Where | Licence / note |
|---|---|---|
| NLLB-200 distilled, 600M parameters | https://huggingface.co/facebook/nllb-200-distilled-600M | CC BY-NC 4.0 (non-commercial) |

## Metric

| What | Where |
|---|---|
| chrF | Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT 2015. https://aclanthology.org/W15-3049/ |
| sacreBLEU (chrF implementation) | Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. https://github.com/mjpost/sacrebleu |

## Libraries and platforms

- PyTorch — https://pytorch.org
- Hugging Face Transformers — https://github.com/huggingface/transformers
- Hugging Face Datasets — https://github.com/huggingface/datasets
- Gradio (demo app) — https://www.gradio.app
- pandas, NumPy, SciPy, Matplotlib — https://pandas.pydata.org, https://numpy.org, https://scipy.org, https://matplotlib.org
- Google Colab (free T4 GPU) — https://colab.research.google.com
- GitHub and GitHub CLI — https://github.com, https://cli.github.com

## AI tools

- **Claude Code** (Anthropic), model Claude Opus 5.5 — https://claude.com/claude-code
  Used to: write the code (`senim.py`, `app.py`, notebook builder, tests), run the notebook in Google Colab through a browser,
  suggest topics for the own test set, give the 1–5 ratings in `data/ratings.csv` as an LLM judge,
  and draft this README/SOURCES. All numbers in the README come from the Colab run, not from the AI.
