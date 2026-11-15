"""Builds space/ : the files to upload to a Hugging Face Space (SDK: Gradio, free CPU).

Run:  python tools/make_space.py
Then upload everything inside space/ to the Space (Files -> Add file -> Upload files).
"""
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "space"
out.mkdir(exist_ok=True)

shutil.copyfile(root / "app.py", out / "app.py")
shutil.copyfile(root / "senim.py", out / "senim.py")
shutil.copyfile(root / "results" / "threshold.json", out / "threshold.json")

# gradio itself is provided by the Space SDK; LaBSE / scikit-learn are only needed for the notebook.
(out / "requirements.txt").write_text(
    "torch\ntransformers>=4.40\nsentencepiece\nsacrebleu>=2.4\n", encoding="utf-8", newline="\n")

readme = """---
title: Senim
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
short_description: English to Kazakh translation that warns when not to trust it
---

# Senim · Сенім

English → Kazakh translation that warns you when it should not be trusted.

Built for the Veritas AI International High School Hackathon 2026 (track AI + Languages, brief 4.4.2).
Code, results, evaluation and sources: https://github.com/arturbaimuratov1-sys/senim

- Translator: NLLB-200 distilled 600M (Meta).
- Trust check: numbers, units, times, dates and negations must match the original, and the Kazakh text is
  translated back to English and compared with the original (threshold calibrated on human ratings).
- On this free CPU server the first request is slow (the model is downloaded and loaded), later ones take a few seconds.
  The second translation by a bigger model is switched off here (it needs a GPU).

**This project is a prototype and is not medical advice.**
"""
(out / "README.md").write_text(readme, encoding="utf-8", newline="\n")

thr = json.loads((out / "threshold.json").read_text(encoding="utf-8"))
print("space/ built:", sorted(p.name for p in out.iterdir()), "| threshold", thr["bt_chrf_threshold"])
