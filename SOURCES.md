# Sources

Everything this project uses, with links. Licences are those stated by each source.

## Data

| What | Where | Licence / note |
|---|---|---|
| FLORES+ (devtest split; `eng_Latn`, `kaz_Cyrl`, `rus_Cyrl`, `fra_Latn`) | https://huggingface.co/datasets/openlanguagedata/flores_plus | CC BY-SA 4.0, gated (terms accepted on Hugging Face). Maintained by the Open Language Data Initiative, https://oldi.org |
| Original FLORES-200 benchmark | NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*, https://arxiv.org/abs/2207.04672 | — |
| `data/own_set.csv`: 50 English sentences (25 medicine, 25 school) | Written by the project author. Claude Code suggested a list of topics; the sentences themselves are the author's. | — |
| `data/ratings.csv`: 1–5 quality ratings of 90 Kazakh translations | **Primary (`rating_1_5`, `human_comment`, `human_correction`): the project author**, rated in Excel (`data/ratings_human_filled.xlsx`). **Secondary (`llm_*` columns): Claude Opus 5.5 as an LLM judge**, rated earlier, kept for comparison (`tools/fill_llm_ratings.py`). | See README, sections 2–3 |

## Model

| What | Where | Licence / note |
|---|---|---|
| NLLB-200 distilled, 600M parameters (main translator) | https://huggingface.co/facebook/nllb-200-distilled-600M | CC BY-NC 4.0 (non-commercial) |
| NLLB-200 distilled, 1.3B parameters (v2: second opinion and back-translation) | https://huggingface.co/facebook/nllb-200-distilled-1.3B | CC BY-NC 4.0 (non-commercial) |
| LaBSE (v2: cross-lingual sentence similarity) | https://huggingface.co/sentence-transformers/LaBSE · Feng et al. (2022), *Language-agnostic BERT Sentence Embedding*, https://arxiv.org/abs/2007.01852 | Apache 2.0 |

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
- CTranslate2 (optional fast inference backend, v4) — https://github.com/OpenNMT/CTranslate2
- edge-tts (text-to-speech for the demo video voice-over; Microsoft Edge neural voices) — https://github.com/rany2/edge-tts
- FFmpeg via imageio-ffmpeg, and Pillow (video assembly and the end card) — https://github.com/imageio/imageio-ffmpeg, https://python-pillow.org
- sentence-transformers — https://www.sbert.net
- scikit-learn (logistic regression, cross-validation, AUC, Cohen's kappa) — https://scikit-learn.org
- pandas, NumPy, SciPy, Matplotlib — https://pandas.pydata.org, https://numpy.org, https://scipy.org, https://matplotlib.org
- Google Colab (free T4 GPU) — https://colab.research.google.com
- Kaggle Notebooks (free T4 GPU; used for the v3 and v4 experiments) — https://www.kaggle.com/code
- GitHub and GitHub CLI — https://github.com, https://cli.github.com

## AI tools

- **Claude Code** (Anthropic), model Claude Opus 5.5 — https://claude.com/claude-code
  Used to: write the code (`senim.py`, `app.py`, notebook builder, tests), run the notebook in Google Colab through a browser,
  suggest topics for the own test set, give the 1–5 ratings in `data/ratings.csv` as an LLM judge,
  run the v3/v4 experiments on Kaggle, and draft README/REPORT/SOURCES.
  Two Claude Code subagents helped in v4: one reviewed the code for bugs, one wrote `senim_fast.py` (CTranslate2 backend) and its tests;
  their work was checked by running it.
  The demo video is the author's screen recording; its voice-over is AI-generated (edge-tts) from a script written by Claude Code and approved by the author, and Claude Code assembled the video (`tools/build_video.py`).
  All numbers come from the Colab and Kaggle runs, not from the AI. Result tables were copied with MD5 checks,
  except `results/v4_speed_backend.csv`, which was copied by hand from the printed benchmark log.
