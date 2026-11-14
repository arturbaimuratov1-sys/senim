"""Senim: English -> Kazakh translation with a trust check.

Shared by notebook.ipynb and app.py.
The trust check has two parts:
  1. Hard checks: numbers, dosage units, months and weekdays in the source
     must survive in the Kazakh translation.
  2. Back-translation: the Kazakh text is translated back to English and
     compared with the source using sentence-level chrF. A low score means
     the meaning probably drifted.
"""
from __future__ import annotations

import re

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_BT_THRESHOLD = 50.0  # used only until a calibrated value exists in results/threshold.json

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None


def load_model(name: str = MODEL_NAME):
    global _model, _tokenizer, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if _device == "cuda" else torch.float32
        _model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype).to(_device).eval()
    return _model, _tokenizer, _device


def translate(texts, src_lang: str, tgt_lang: str, batch_size: int = 16,
              num_beams: int = 4, max_new_tokens: int = 256):
    """Translate a list of strings with NLLB. Language codes are FLORES-200 codes, e.g. eng_Latn."""
    import torch

    model, tok, device = load_model()
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches
    out = [None] * len(texts)
    tok.src_lang = src_lang
    bos = tok.convert_tokens_to_ids(tgt_lang)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            # max_length (not max_new_tokens) overrides the model's generation_config without a warning;
            # for an encoder-decoder it limits only the generated output.
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=num_beams,
                                 max_length=max_new_tokens)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d
    return out


# ---------------------------------------------------------------------------
# Hard checks: numbers, units, dates
# ---------------------------------------------------------------------------

_EN_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19, "once": 1, "twice": 2, "thrice": 3,
             # "first"/"second" are left out on purpose: usually not quantities
             # ("first aid", "first cases", "a second opinion").
             "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}
_EN_MULT = {"hundred": 100, "thousand": 1000}

_KK_UNITS = {"бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7,
             "сегіз": 8, "тоғыз": 9}
_KK_TENS = {"он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50, "алпыс": 60,
            "жетпіс": 70, "сексен": 80, "тоқсан": 90}
_KK_MULT = {"жүз": 100, "мың": 1000}
# Kazakh suffixes allowed after a number word (ordinal, distributive, possessive, case endings).
_KK_SUFFIXES = ("", "інші", "ыншы", "ншы", "нші", "ден", "дан", "тен", "тан", "еу", "ау",
                "ге", "ға", "ке", "қа", "де", "да", "те", "та", "нің", "ның", "дің", "дың",
                "і", "ы", "сі", "сы", "іне", "ына", "ден", "нен", "нан")
# "он" (10) + a vowel/locative suffix collides with common words ("оны" = him, "онда" = there).
_KK_SUFFIXES_ON = ("", "ыншы", "ға", "нан", "нің")

_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Clock times: "6:00 PM", "9:30 am", "18:00", "7.15". A dot only counts as a time separator
# next to AM/PM (English) or when the source has times (Kazakh "7.15-де"); otherwise
# "3.50 m" is a decimal number.
_TIME_COLON_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?![\d])", re.IGNORECASE)
_TIME_DOT_AMPM_RE = re.compile(r"\b(\d{1,2})\.(\d{2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", re.IGNORECASE)
_TIME_ANY_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})()(?![\d])")
_AMPM_RE = re.compile(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", re.IGNORECASE)


def _times(text: str, allow_ampm_without_minutes: bool, dot_times: bool = False):
    """Return (list of acceptable (h, m) forms per time, text with times removed)."""
    found = []

    def forms(h, m, ampm):
        h, m = int(h), int(m)
        if ampm:
            pm = ampm.lower().startswith("p")
            h24 = (h % 12) + (12 if pm else 0)
            return {(h24, m), (h, m)}  # "6:00 PM" may be written as 18:00 or 6:00
        return {(h, m)}

    def sub_time(mt):
        found.append(forms(mt.group(1), mt.group(2), mt.group(3)))
        return " "

    if dot_times:
        text = _TIME_ANY_RE.sub(sub_time, text)
    else:
        text = _TIME_COLON_RE.sub(sub_time, text)
        text = _TIME_DOT_AMPM_RE.sub(sub_time, text)
    if allow_ampm_without_minutes:
        def sub_ampm(mt):
            found.append(forms(mt.group(1), 0, mt.group(2)))
            return " "
        text = _AMPM_RE.sub(sub_ampm, text)
    return found, text


def _norm_num(s: str) -> float:
    return float(s.replace(",", "."))


def _digit_numbers(text: str):
    return [_norm_num(m) for m in _DIGIT_RE.findall(text)]


def _parse_words(tokens, units, tens, mult):
    """Turn a token stream into numbers by summing consecutive number words.
    A token ending with "|" closes the number (a Kazakh case suffix ends a compound:
    "бірден қырыққа" is 1 and 40, not 41)."""
    nums, cur, active = [], 0, False
    for t in tokens:
        closes = t.endswith("|")
        t = t.rstrip("|")
        if closes and t in {**units, **tens, **mult}:
            if t in mult:
                cur = (cur or 1) * mult[t]
            else:
                cur += {**units, **tens}[t]
            nums.append(float(cur))
            cur, active = 0, False
            continue
        if t in units:
            cur += units[t]; active = True
        elif t in tens:
            cur += tens[t]; active = True
        elif t in mult:
            cur = (cur or 1) * mult[t]; active = True
        else:
            if active:
                nums.append(float(cur))
            cur, active = 0, False
    if active:
        nums.append(float(cur))
    return nums


def _en_word_numbers(text: str):
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    return _parse_words(tokens, _EN_UNITS, _EN_TENS, _EN_MULT)


def _kk_token_value(tok: str):
    """Map a Kazakh token to its number word; append "|" if it carries a suffix."""
    for table in (_KK_UNITS, _KK_TENS, _KK_MULT):
        for word in table:
            # Final consonant alternation before a vowel suffix: қырық -> қырығы.
            stems = {word}
            if word[-1] in "кқп":
                stems.add(word[:-1] + {"к": "г", "қ": "ғ", "п": "б"}[word[-1]])
            for stem in stems:
                if not tok.startswith(stem):
                    continue
                rest = tok[len(stem):]
                allowed = _KK_SUFFIXES_ON if word == "он" else _KK_SUFFIXES
                if rest in allowed:
                    return word + ("|" if rest else "")
    return None


def _kk_word_numbers(text: str):
    tokens = re.findall(r"[а-яәғқңөұүһіё]+", text.lower())
    mapped = [(_kk_token_value(t) or "#") for t in tokens]
    return _parse_words(mapped, _KK_UNITS, _KK_TENS, _KK_MULT)


def check_numbers(src: str, tgt: str):
    """Every number in the source must appear in the translation, and every digit
    number in the translation must come from the source. Number words count
    ("two" == "екі" == 2), but Kazakh 'бір' alone is not counted as an extra number
    because it is also the indefinite article."""
    reasons = []
    # Clock times are compared as times first, then removed from the text.
    src_times, src = _times(src, allow_ampm_without_minutes=True)
    tgt_times, tgt = _times(tgt, allow_ampm_without_minutes=False, dot_times=bool(src_times))
    tgt_time_set = set().union(*tgt_times) if tgt_times else set()
    src_time_set = set().union(*src_times) if src_times else set()
    bad_src = [f for f in src_times if not (f & tgt_time_set)]
    bad_tgt = [f for f in tgt_times if not (f & src_time_set)]
    if bad_src:
        reasons.append("time missing or changed: " + ", ".join(
            "%d:%02d" % min(f) for f in bad_src))
    if bad_tgt:
        reasons.append("time not in the original: " + ", ".join(
            "%d:%02d" % min(f) for f in bad_tgt))

    src_nums = _digit_numbers(src) + _en_word_numbers(src)
    tgt_digits = _digit_numbers(tgt)
    tgt_all = tgt_digits + _kk_word_numbers(tgt)
    missing = sorted({n for n in src_nums if n not in tgt_all})
    extra = sorted({n for n in tgt_digits if n not in src_nums})
    fmt = lambda xs: ", ".join(str(int(x)) if x.is_integer() else str(x) for x in xs)
    if missing:
        reasons.append(f"number missing or changed: {fmt(missing)}")
    if extra:
        reasons.append(f"number not in the original: {fmt(extra)}")
    return reasons


_UNITS = {
    "mg": (r"\b(mg|milligrams?)\b", ("мг", "mg", "миллиграмм")),
    "ml": (r"\b(ml|millilit(er|re)s?)\b", ("мл", "ml", "миллилитр")),
    "mcg": (r"(\bmcg\b|µg|\bmicrograms?\b)", ("мкг", "mcg", "µg", "микрограмм")),
    "kg": (r"\b(kg|kilograms?)\b", ("кг", "kg", "килограмм")),
    "%": (r"%|\bpercent\b", ("%", "пайыз")),
    "°C": (r"°\s?c\b|\bdegrees?\b", ("°", "градус")),
}


def check_units(src: str, tgt: str):
    reasons = []
    s, t = src.lower(), tgt.lower()
    for name, (pattern, kk_forms) in _UNITS.items():
        if re.search(pattern, s) and not any(f in t for f in kk_forms):
            reasons.append(f"dosage unit missing: {name}")
    return reasons


# Regex stems; "қыркүйе[кг]" covers the к->г alternation ("қыркүйегінде").
# Weekdays include the common colloquial forms (шейшембі, шаршамбы, бейшембі).
_MONTHS = {"january": "қаңтар", "february": "ақпан", "march": "наурыз", "april": "сәуір",
           "may": "мамыр", "june": "маусым", "july": "шілде", "august": "тамыз",
           "september": "қыркүйе[кг]", "october": "қазан", "november": "қараша",
           "december": "желтоқсан"}
_WEEKDAYS = {"monday": "дүйсенбі", "tuesday": "(сейсенбі|шейшембі)",
             "wednesday": "(сәрсенбі|шаршамбы)", "thursday": "(бейсенбі|бейшембі)",
             "friday": "жұма", "saturday": "сенбі", "sunday": "жексенбі"}


def check_dates(src: str, tgt: str):
    reasons = []
    t = tgt.lower()
    for en, kk in {**_MONTHS, **_WEEKDAYS}.items():
        if en == "may":
            # Only the month: capitalised "May" next to a number ("May 5", "5 May").
            found = re.search(r"(\d\s*May\b|\bMay\s*\d)", src)
        else:
            found = re.search(rf"\b{en}\b", src.lower())
        if found and not re.search(rf"\b{kk}", t):
            reasons.append(f"date word changed or missing: {en.capitalize()}")
    return reasons


def hard_checks(src: str, tgt: str):
    return check_numbers(src, tgt) + check_units(src, tgt) + check_dates(src, tgt)


# ---------------------------------------------------------------------------
# Back-translation score and verdict
# ---------------------------------------------------------------------------

def sentence_chrf(hyp: str, ref: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def verdict(src: str, tgt: str, back: str, threshold: float = DEFAULT_BT_THRESHOLD):
    """Return a dict with the trust label and the reasons behind it."""
    reasons = hard_checks(src, tgt)
    bt = sentence_chrf(back, src)
    if bt < threshold:
        reasons.append(f"back-translation differs from the original (chrF {bt:.0f} < {threshold:.0f})")
    return {"trust": not reasons, "reasons": reasons, "bt_chrf": bt, "back_translation": back}


# Plain-language reason texts shown to a Kazakh-speaking user.
KK_REASON = {
    "number": "Сандар түпнұсқамен сәйкес келмейді.",
    "time": "Уақыт (сағат) түпнұсқамен сәйкес келмейді.",
    "dosage": "Дәрі мөлшерінің бірлігі (мг, мл) жоғалып кетті.",
    "date": "Күн немесе ай атауы өзгерді.",
    "back-translation": "Кері аударма түпнұсқаның мағынасынан алыс.",
}


def kk_reasons(reasons):
    out = []
    for r in reasons:
        for key, text in KK_REASON.items():
            if r.startswith(key) or key in r.split(":")[0]:
                if text not in out:
                    out.append(text)
    return out
