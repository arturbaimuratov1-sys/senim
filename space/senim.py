"""Senim: English -> Kazakh translation with a trust check.

Shared by notebook.ipynb and app.py.
The trust check has two parts:
  1. Hard checks: numbers, dosage units, months and weekdays in the source
     must survive in the Kazakh translation.
  2. Back-translation: the Kazakh text is translated back to English and
     compared with the source using sentence-level chrF. A low score means
     the meaning probably drifted.
v2 adds: a negation check, a second opinion from a bigger model (NLLB 1.3B),
back-translation by that bigger model, LaBSE meaning similarity, and a small
logistic model that combines all signals (trained in notebook.ipynb).
"""
from __future__ import annotations

import os
import re
import threading

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_BT_THRESHOLD = 50.0  # used only until a calibrated value exists in results/threshold.json

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

BIG_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
LABSE_NAME = "sentence-transformers/LaBSE"

_models = {}  # name -> (model, tokenizer, device)
# The shared tokenizer keeps src_lang as state: hold this lock from setting src_lang to the end of generate,
# so two concurrent requests (e.g. Gradio with concurrency > 1) cannot mix their languages.
_LOCK = threading.RLock()


def _use_ct2() -> bool:
    """SENIM_BACKEND=ct2 switches translate/translate_n to the CTranslate2 int8 backend (senim_fast.py)."""
    return os.environ.get("SENIM_BACKEND", "").lower() == "ct2"


def load_model(name: str = MODEL_NAME):
    if name not in _models:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        _models[name] = (model, tok, device)
    return _models[name]


def translate(texts, src_lang: str, tgt_lang: str, batch_size: int = 16,
              num_beams: int = 4, max_new_tokens: int = 256, model_name: str = MODEL_NAME):
    """Translate a list of strings with NLLB. Language codes are FLORES-200 codes, e.g. eng_Latn.
    max_new_tokens is passed to generate() as max_length (see the comment below)."""
    if _use_ct2():
        import senim_fast
        return senim_fast.translate(texts, src_lang, tgt_lang, batch_size=batch_size, num_beams=num_beams,
                                    max_new_tokens=max_new_tokens, model_name=model_name)
    import torch

    model, tok, device = load_model(model_name)
    texts = _as_list(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches
    out = [None] * len(texts)
    bos = tok.convert_tokens_to_ids(tgt_lang)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        with _LOCK, torch.inference_mode():
            tok.src_lang = src_lang
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            # max_length (not max_new_tokens) overrides the model's generation_config without a warning;
            # for an encoder-decoder it limits only the generated output.
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=num_beams,
                                 max_length=max_new_tokens)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d
    return out


def _as_list(texts):
    """Accept one string or a list of strings."""
    if isinstance(texts, str):
        return [texts]
    return [str(t) for t in texts]


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

# "1,000" / "1 000" / "2 500" are one number (thousands separator); "3,50" or "3.5" is a decimal.
_DIGIT_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[ ,\u00a0]\d{3})+(?![\d.,]\d)|\d+(?:[.,]\d+)?")
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
    s = s.replace("\u00a0", "").replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):  # thousands separator
        return float(s.replace(",", ""))
    return float(s.replace(",", "."))


def _digit_numbers(text: str):
    return [_norm_num(m) for m in _DIGIT_RE.findall(text)]


def _parse_words(tokens, units, tens, mult):
    """Turn a token stream into numbers ("two thousand five hundred" -> 2500).
    A token ending with "|" closes the number (a Kazakh case suffix ends a compound:
    "бірден қырыққа" is 1 and 40, not 41)."""
    nums, total, cur, active = [], 0, 0, False

    def flush():
        nonlocal total, cur, active
        if active:
            nums.append(float(total + cur))
        total, cur, active = 0, 0, False

    for t in tokens:
        closes = t.endswith("|")
        t = t.rstrip("|")
        if t in units or t in tens:
            cur += units.get(t, 0) + tens.get(t, 0)
        elif t in mult:
            if mult[t] >= 1000:          # thousand closes a group: total += group * 1000
                total += (cur or 1) * mult[t]
                cur = 0
            else:                         # hundred multiplies only the current group
                cur = (cur or 1) * mult[t]
        else:
            flush()
            continue
        active = True
        if closes:
            flush()
    flush()
    return nums


def _en_word_numbers(text: str):
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    words = {**_EN_UNITS, **_EN_TENS, **_EN_MULT}
    # "one hundred and five": "and" between number words is part of the number
    tokens = [t for i, t in enumerate(tokens)
              if not (t == "and" and 0 < i < len(tokens) - 1 and tokens[i - 1] in words and tokens[i + 1] in words)]
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
    # "6 pm" has no minutes: Kazakh often writes just the hour ("сағат 18-де"), so a bare number equal to
    # the hour (12- or 24-hour form) also counts, and that number is then not an "extra" number.
    tgt_bare = set(_digit_numbers(tgt))
    hour_hits, still_bad = set(), []
    for f in bad_src:
        hit = next((float(h) for h, m in sorted(f) if m == 0 and float(h) in tgt_bare), None)
        if hit is None:
            still_bad.append(f)
        else:
            hour_hits.add(hit)
    bad_src = still_bad
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
    extra = sorted({n for n in tgt_digits if n not in src_nums and n not in hour_hits})
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
        elif en == "march":
            # "march" is also a verb: only the capitalised word counts as the month.
            found = re.search(r"\bMarch\b", src)
        else:
            found = re.search(rf"\b{en}\b", src.lower())
        if found and not re.search(rf"\b{kk}", t):
            reasons.append(f"date word changed or missing: {en.capitalize()}")
    return reasons


def hard_checks(src: str, tgt: str):
    return check_numbers(src, tgt) + check_units(src, tgt) + check_dates(src, tgt)


# ---------------------------------------------------------------------------
# Negation check (v2): "if you notice bleeding" must not become "if you do NOT ..."
# ---------------------------------------------------------------------------

# English: grammatical negation plus words that Kazakh normally renders with a negative verb
# ("failed to finish" -> аяқтай алмады, "despite" -> қарамастан, "avoid" -> болдырмау).
# The prefix un- is NOT counted: "unusual bleeding" is correctly "әдеттен тыс", without negation.
_EN_NEG_RE = re.compile(
    r"\b(not|never|no|none|nothing|nobody|neither|nor|without|unless|cannot|"
    r"unable|fail(?:s|ed|ing)?|avoid(?:s|ed|ing)?|despite|instead|"
    r"ban(?:s|ned)?|forbid(?:s|den)?|prohibit(?:s|ed)?|refus(?:e|es|ed|ing)|lack(?:s|ed|ing)?|seldom|rarely|"
    r"invisible|obsolete|untapped)\b|n't\b", re.IGNORECASE)
# Kazakh negation: separate words, and the verbal negative suffix -ма/-ме/-ба/-бе/-па/-пе
# followed by a typical verb ending (imperative, conditional, participle, verbal noun, gerund).
_KK_NEG_WORDS_RE = re.compile(r"(?<![а-яәғқңөұүһі])(емес|жоқ|ешқашан|ешбір|ештеңе|ешкім|ешқандай|ешқайда)")
_KK_NEG_VERB_RE = re.compile(
    r"(?<![а-яәғқңөұүһі])([а-яәғқңөұүһі]{2,}(?:ма|ме|ба|бе|па|пе))"
    r"(ңыз|ңіз|ңдар|ңдер|са|се|саңыз|сеңіз|сақ|сек|ған|ген|қан|кен|ды|ді|йды|йді|йтын|йтін|у|уы|уі|уға|уге|"
    r"ймын|ймін|йсыз|йсіз|йміз|ймыз|стан|стен|й|йынша|йінше|ғандықтан|гендіктен)"
    r"(?![а-яәғқңөұүһі])")


# Nouns whose stem already ends in -ма/-ме/-ба/-бе/-па/-пе: "бөлмеңіз" = "your room", not a negative verb.
_KK_NOUN_STEMS = ("бөлме", "анықтама", "бағдарлама", "хабарлама", "кеме", "түйме", "қаптама", "нұсқама",
                  "тізбе", "арба")  # only unambiguous nouns; e.g. "тапсырмаңыз" can also mean "do not hand in"


def _kk_negative_verbs(t: str) -> int:
    n = 0
    for m in _KK_NEG_VERB_RE.finditer(t):
        if not m.group(1).endswith(_KK_NOUN_STEMS):
            n += 1
    return n


def negation_counts(src: str, tgt: str):
    en = len(_EN_NEG_RE.findall(src))
    t = tgt.lower()
    kk = len(_KK_NEG_WORDS_RE.findall(t)) + _kk_negative_verbs(t)
    return en, kk


def check_negation(src: str, tgt: str):
    """Flag when the number of negations differs between source and translation."""
    en, kk = negation_counts(src, tgt)
    if en == kk:
        return []
    if kk > en:
        return [f"negation added: {kk} in translation vs {en} in original"]
    return [f"negation lost: {kk} in translation vs {en} in original"]


# ---------------------------------------------------------------------------
# Back-translation score and verdict
# ---------------------------------------------------------------------------

def sentence_chrf(hyp: str, ref: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def verdict(src: str, tgt: str, back: str, threshold: float = DEFAULT_BT_THRESHOLD, negation: bool = True):
    """Return a dict with the trust label and the reasons behind it.
    negation=True adds the v2 negation check (used by the app; see README, section 6)."""
    reasons = hard_checks(src, tgt) + (check_negation(src, tgt) if negation else [])
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
    "negation": "«Емес / жоқ / -ма, -ме» түпнұсқадағыдай емес: мағынасы керісінше болуы мүмкін.",
}


def kk_reasons(reasons):
    out = []
    for r in reasons:
        for key, text in KK_REASON.items():
            if r.startswith(key) or key in r.split(":")[0]:
                if text not in out:
                    out.append(text)
    return out


# ---------------------------------------------------------------------------
# v2 signals and combined score
# ---------------------------------------------------------------------------

_labse = None


def labse_similarity(src_texts, tgt_texts):
    """Cosine similarity of LaBSE sentence embeddings (English source vs Kazakh translation)."""
    global _labse
    if _labse is None:
        from sentence_transformers import SentenceTransformer
        _labse = SentenceTransformer(LABSE_NAME)
    a = _labse.encode(list(src_texts), normalize_embeddings=True, batch_size=32)
    b = _labse.encode(list(tgt_texts), normalize_embeddings=True, batch_size=32)
    return [float((x * y).sum()) for x, y in zip(a, b)]


V2_FEATURES = ["bt_chrf", "bt_big_chrf", "agree_chrf", "labse", "hard_flag", "neg_flag"]


def v2_signals(src_texts, mt_texts):
    """All v2 signals for (English source, Kazakh translation from the 600M model) pairs."""
    src_texts, mt_texts = list(src_texts), list(mt_texts)
    back = translate(mt_texts, "kaz_Cyrl", "eng_Latn")
    back_big = translate(mt_texts, "kaz_Cyrl", "eng_Latn", model_name=BIG_MODEL_NAME)
    mt_big = translate(src_texts, "eng_Latn", "kaz_Cyrl", model_name=BIG_MODEL_NAME)
    sims = labse_similarity(src_texts, mt_texts)
    rows = []
    for s_, t_, b_, bb, mb, sim in zip(src_texts, mt_texts, back, back_big, mt_big, sims):
        hard = hard_checks(s_, t_)
        neg = check_negation(s_, t_)
        rows.append({"back_en": b_, "back_en_big": bb, "mt_big": mb,
                     "bt_chrf": sentence_chrf(b_, s_), "bt_big_chrf": sentence_chrf(bb, s_),
                     "agree_chrf": sentence_chrf(t_, mb), "labse": sim,
                     "hard_flag": float(bool(hard)), "neg_flag": float(bool(neg)),
                     "reasons": hard + neg})
    return rows


def v2_risk(features: dict, model: dict) -> float:
    """Probability that the translation is bad, from the logistic model saved by the notebook."""
    import math
    z = model["intercept"]
    for name, coef, mu, sd in zip(model["features"], model["coef"], model["mean"], model["std"]):
        z += coef * (features[name] - mu) / (sd or 1.0)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# v3: reduce errors instead of only flagging them
#   - translate_n:       several candidate translations (beam search, n best)
#   - translate_split:   translate sentence by sentence (fights dropped sentences)
#   - select_best:       pick the candidate that passes the checks and back-translates best
# ---------------------------------------------------------------------------

def translate_n(texts, src_lang: str, tgt_lang: str, n: int = 4, num_beams: int = 4,
                model_name: str = MODEL_NAME, batch_size: int = 8, max_new_tokens: int = 256):
    """Return the n best beam-search translations for every text (list of lists).
    The first candidate is the same beam search as translate(...); it can differ in rare cases because
    batches are padded differently in fp16."""
    if _use_ct2():
        import senim_fast
        return senim_fast.translate_n(texts, src_lang, tgt_lang, n=n, num_beams=num_beams, model_name=model_name,
                                      batch_size=batch_size, max_new_tokens=max_new_tokens)
    import torch

    model, tok, device = load_model(model_name)
    texts = _as_list(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    bos = tok.convert_tokens_to_ids(tgt_lang)
    beams = max(num_beams, n)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        with _LOCK, torch.inference_mode():
            tok.src_lang = src_lang
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams,
                                 num_return_sequences=n, max_length=max_new_tokens)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j, i in enumerate(idx):
            out[i] = dec[j * n:(j + 1) * n]
    return out


_ABBR = {"mr", "mrs", "ms", "dr", "st", "no", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "jr", "sr",
         "prof", "inc", "ltd", "co", "mt", "approx", "fig", "a.m", "p.m", "dept", "gen", "gov", "sen"}
_SENT_END = re.compile(r"[.!?][\"”’)]?\s+(?=[\"“(]?[A-Z0-9])")


def split_sentences(text: str):
    """Split English text into sentences; abbreviations (U.S., Dr.) and initials (George W. Bush) do not split."""
    parts, start = [], 0
    for m in _SENT_END.finditer(text):
        chunk = text[start:m.start() + 1].split()
        last = chunk[-1].rstrip(".").lower() if chunk else ""
        if last in _ABBR or (len(last) == 1 and last.isalpha()):
            continue
        parts.append(text[start:m.end()].strip())
        start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def translate_split(texts, src_lang: str, tgt_lang: str, **kw):
    """Translate every sentence separately and join them again."""
    texts = _as_list(texts)
    pieces = [split_sentences(t) or [t] for t in texts]
    flat = [p for ps in pieces for p in ps]
    tr = translate(flat, src_lang, tgt_lang, **kw)
    out, k = [], 0
    for ps in pieces:
        out.append(" ".join(tr[k:k + len(ps)]))
        k += len(ps)
    return out


def select_best(srcs, candidates, back=None):
    """For each source pick the candidate that (1) passes the hard checks and the negation check,
    then (2) has the highest back-translation chrF. Ties keep the earlier candidate (= the default output).
    candidates: list of lists of Kazakh strings. back: optional dict {kazakh: english back-translation}.
    Returns (chosen translations, chosen indices, back dict)."""
    back = dict(back or {})
    todo = sorted({c for cs in candidates for c in cs if c not in back})
    if todo:
        for k, b in zip(todo, translate(todo, "kaz_Cyrl", "eng_Latn")):
            back[k] = b
    chosen, idxs = [], []
    for src, cs in zip(srcs, candidates):
        def score(i):
            c = cs[i]
            penalty = 1000 * bool(hard_checks(src, c)) + 1000 * bool(check_negation(src, c))
            return (-penalty, sentence_chrf(back[c], src), -i)
        best = max(range(len(cs)), key=score)
        chosen.append(cs[best])
        idxs.append(best)
    return chosen, idxs, back


def improved_translate(texts, src_lang: str = "eng_Latn", tgt_lang: str = "kaz_Cyrl"):
    """System R (v3): the best of 8 candidates (4 from NLLB 600M + 4 from NLLB 1.3B), picked by select_best.
    On FLORES+ dev it scored +2.15 chrF over the plain 600M output; see experiments/P3_PLAN.md and REPORT.md."""
    texts = _as_list(texts)
    c600 = translate_n(texts, src_lang, tgt_lang, n=4, num_beams=4)
    c13 = translate_n(texts, src_lang, tgt_lang, n=4, num_beams=4, model_name=BIG_MODEL_NAME)
    chosen, _, _ = select_best(texts, [a + b for a, b in zip(c600, c13)])
    return chosen


# ---------------------------------------------------------------------------
# v4: smart cascade — fast for safe sentences, best-of-8 only for risky ones
# ---------------------------------------------------------------------------

def back_translate_cached(texts, back=None, model_name: str = MODEL_NAME):
    """Back-translate Kazakh -> English, reusing (and filling) the dict `back`. Duplicates are translated once."""
    back = {} if back is None else back
    todo = sorted({t for t in texts if t not in back})
    if todo:
        for k, b in zip(todo, translate(todo, "kaz_Cyrl", "eng_Latn", model_name=model_name)):
            back[k] = b
    return back


def is_safe(src: str, tgt: str, back_en: str, threshold: float) -> bool:
    """The trust check of the app, as a yes/no: no number/date/negation problem and back-translation close enough."""
    return not hard_checks(src, tgt) and not check_negation(src, tgt) and sentence_chrf(back_en, src) >= threshold


def smart_translate(texts, threshold: float = DEFAULT_BT_THRESHOLD, src_lang: str = "eng_Latn",
                    tgt_lang: str = "kaz_Cyrl"):
    """Cascade (v4).
    Stage 1, every sentence: NLLB 600M (beam 4; the 4 beams come for free) + back-translation of the top beam.
            If it passes the trust check, keep it.
    Stage 2, only risky sentences: add the 4 best beams of NLLB 1.3B and pick the best of 8 with select_best
            (exactly system R of v3), reusing stage 1's candidates and back-translations.
    Returns (translations, stages, back) where stages[i] is "fast" or "best_of_8"."""
    texts = _as_list(texts)
    c600 = translate_n(texts, src_lang, tgt_lang, n=4, num_beams=4)
    top = [c[0] for c in c600]
    back = back_translate_cached(top)
    out, stages = list(top), ["fast"] * len(texts)
    risky = [i for i, (s, t) in enumerate(zip(texts, top)) if not is_safe(s, t, back[t], threshold)]
    if risky:
        c13 = translate_n([texts[i] for i in risky], src_lang, tgt_lang, n=4, num_beams=4, model_name=BIG_MODEL_NAME)
        cands = [c600[i] + c for i, c in zip(risky, c13)]
        chosen, _, back = select_best([texts[i] for i in risky], cands, back)
        for i, c in zip(risky, chosen):
            out[i], stages[i] = c, "best_of_8"
    return out, stages, back
