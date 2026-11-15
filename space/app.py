"""Senim demo: English -> Kazakh translation with a trust label.

Run:  pip install -r requirements.txt  &&  python app.py
(In Colab: !python app.py, Gradio prints a public link.)
This project is a prototype and is not medical advice.
"""
import json
import os

import gradio as gr

import senim

THRESHOLD = senim.DEFAULT_BT_THRESHOLD
THRESHOLD_NOTE = "default value (not calibrated yet)"
for _path in ("results/threshold.json", "threshold.json"):  # the second one is used on Hugging Face Spaces
    if os.path.exists(_path):
        with open(_path, encoding="utf-8") as f:
            THRESHOLD = float(json.load(f)["bt_chrf_threshold"])
        THRESHOLD_NOTE = f"calibrated on human ratings ({_path})"
        break

# The second translation (NLLB 1.3B) needs a GPU to be usable: it is shown only when a GPU is present,
# or when SENIM_SECOND=1 is set. On a free CPU server it is switched off.
import torch  # noqa: E402

ENABLE_SECOND = torch.cuda.is_available() or os.environ.get("SENIM_SECOND") == "1"


def run(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "", ""
    # One beam search gives both the translation (top beam) and 3 spare candidates for the best-of-8 step,
    # so nothing is translated twice; the back-translation is cached and reused there as well.
    c600 = senim.translate_n([text], "eng_Latn", "kaz_Cyrl", n=4, num_beams=4)[0]
    kk = c600[0]
    back_cache = senim.back_translate_cached([kk])
    back = back_cache[kk]
    v = senim.verdict(text, kk, back, THRESHOLD, negation=True)
    second = ""
    if v["trust"]:
        label = ("### ✅ Аударма сенімді көрінеді\n"
                 "Likely OK: numbers, units, dates and negations match, "
                 "and the back-translation is close to the original.\n\n"
                 "Маңызды шешім алдында (дәрі, мектеп) бәрібір ағылшынша білетін адамнан сұраңыз.")
    else:
        kk_lines = "\n".join(f"- {r}" for r in senim.kk_reasons(v["reasons"]))
        en_lines = "\n".join(f"- {r}" for r in v["reasons"])
        label = ("### ⚠️ Бұл аудармаға сенуге болмайды\n"
                 "Do not rely on this translation. Ask a person who reads English.\n\n"
                 f"{kk_lines}\n\n**Why (details):**\n{en_lines}")
        # Improved translation (v3, system R): the best of 8 candidates from NLLB 600M and 1.3B,
        # picked to pass the number/date/negation checks and to back-translate best. Not re-checked here,
        # because the trust threshold was calibrated on the plain 600M output.
        if ENABLE_SECOND:
            c13 = senim.translate_n([text], "eng_Latn", "kaz_Cyrl", n=4, num_beams=4,
                                    model_name=senim.BIG_MODEL_NAME)[0]
            second = senim.select_best([text], [c600 + c13], back_cache)[0][0]
            if senim.hard_checks(text, second) or senim.check_negation(text, second):
                second += ("\n\n⚠️ Бұл нұсқа да тексеруден өтпеді / this version also fails the checks "
                           "(numbers, dates or negation). Ask a person who reads English.")
    details = f"Back-translation chrF: {v['bt_chrf']:.1f} (threshold {THRESHOLD:.1f}, {THRESHOLD_NOTE})"
    return kk, label, second, f"{back}\n\n{details}"


EXAMPLES = [
    "Take one 500 mg tablet twice a day after meals for 7 days.",
    "Do not give this medicine to children under 12 years old.",
    "The parent-teacher meeting will be held on Thursday, March 5, at 6 pm in room 214.",
    "Please return the signed permission slip by Friday.",
    "Stop taking the tablets and call your doctor if you notice unusual bleeding.",
]

with gr.Blocks(title="Senim") as demo:
    gr.Markdown("# Senim · Сенім\nEnglish → Kazakh translation that warns you when it should not be trusted.\n\n"
                "*This project is a prototype and is not medical advice.*")
    with gr.Row():
        with gr.Column():
            src = gr.Textbox(label="English text / Ағылшынша мәтін", lines=6)
            btn = gr.Button("Translate / Аудару", variant="primary")
            gr.Examples(EXAMPLES, inputs=src)
        with gr.Column():
            out = gr.Textbox(label="Kazakh / Қазақша", lines=6)
            trust = gr.Markdown()
            second = gr.Textbox(label="Improved translation: best of 8 candidates (only with a warning) / "
                                      "Жақсартылған аударма: 8 нұсқаның ең жақсысы", lines=4, visible=ENABLE_SECOND)
            back = gr.Textbox(label="Back-translation to English (for checking)", lines=4)
    btn.click(run, inputs=src, outputs=[out, trust, second, back])

if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("COLAB_RELEASE_TAG")))
