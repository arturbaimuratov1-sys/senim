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
if os.path.exists("results/threshold.json"):
    with open("results/threshold.json", encoding="utf-8") as f:
        THRESHOLD = float(json.load(f)["bt_chrf_threshold"])
    THRESHOLD_NOTE = "calibrated on human ratings (results/threshold.json)"


def run(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "", ""
    kk = senim.translate([text], "eng_Latn", "kaz_Cyrl")[0]
    back = senim.translate([kk], "kaz_Cyrl", "eng_Latn")[0]
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
        # Second opinion from the bigger model: more accurate on FLORES+, but not checked by Senim.
        second = senim.translate([text], "eng_Latn", "kaz_Cyrl", model_name=senim.BIG_MODEL_NAME)[0]
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
            second = gr.Textbox(label="Second translation by a bigger model (only with a warning; not checked) / "
                                      "Екінші нұсқа (үлкен модель, тексерілмеген)", lines=4)
            back = gr.Textbox(label="Back-translation to English (for checking)", lines=4)
    btn.click(run, inputs=src, outputs=[out, trust, second, back])

if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("COLAB_RELEASE_TAG")))
