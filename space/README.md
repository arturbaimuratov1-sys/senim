---
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
