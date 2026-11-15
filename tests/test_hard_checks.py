"""Tests for the rule-based part of the trust check (no model needed).

Run:  python -m pytest tests/
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import senim  # noqa: E402

# (source, translation, should_flag)
CASES = [
    ("Take two 500 mg tablets twice a day until 5 May.",
     "Күніне екі рет 500 мг-ден екі таблетка 5 мамырға дейін ішіңіз.", False),
    ("Take 250 mg on Monday.", "Дүйсенбіде 500 мг ішіңіз.", True),          # changed dose
    ("Take 5 ml.", "Бір қасық 5 ішіңіз.", True),                              # unit lost
    ("School starts on Saturday, March 3.", "Мектеп 3 наурыз, жексенбіде басталады.", True),
    ("Give fifteen drops.", "Он бес тамшы беріңіз.", False),                  # number words
    ("Meeting on Tuesday at 6:00 PM", "шейшембі, сағат 18:00-де", False),     # 12h -> 24h, colloquial day
    ("Meeting at 6:00 PM", "сағат 19:00-де", True),                           # wrong time
    ("pickup at 7:15 AM", "сағат 7.15-де", False),
    ("pickup at 7:15 AM", "сағат 7.25-де", True),
    ("Between 10:00-11:00 pm MDT, a fire", "сағат 10:00 - 11:00 аралығында", False),
    ("the figure is one to forty", "бұл көрсеткіш бірден қырыққа дейін", False),
    ("the first cases", "алғашқы жағдайлары", False),                          # "first" is not a quantity
    ("one of many pyramids", "көптеген пирамидалардың бірі", False),
    ("in September 2017", "2017 жылдың қыркүйегінде", False),                   # к -> г alternation
    ("closed all of Wednesday", "шаршамбы күні", False),
    ("it is 3.50 m wide", "ені 3.50 метр", False),                             # decimal, not a time
    ("6 inches of rainfall", "жауын-шашынның 10 сантиметрі", True),
    ("Take 10 ml.", "Оны 5 мл ішіңіз.", True),                                # "оны" is not 10
    ("Wait ten days.", "Оны күтіңіз.", True),
]


@pytest.mark.parametrize("src,tgt,should_flag", CASES)
def test_hard_checks(src, tgt, should_flag):
    assert bool(senim.hard_checks(src, tgt)) == should_flag, senim.hard_checks(src, tgt)


NEG_CASES = [
    ("Stop the tablets if you notice unusual bleeding.",
     "Егер сізде әдеттегідей қан кетпесе, таблеткаларды тоқтатыңыз.", True),   # negation added
    ("If you have not paid the fee by Monday your child may miss the trip.",
     "Егер сіз дүйсенбіге дейін ақыны төлемесеңіз, балаңыз сапарды жіберіп алмауы мүмкін.", True),
    ("Do not give this medicine to children under 2 years.",
     "Бұл препаратты 2 жасқа дейінгі балаларға бермеңіз.", False),
    ("Take the medicine after meals.", "Дәріні тамақтан кейін қабылдаңыз.", False),
    ("Keep the bottle in a cool place.", "Флаконды салқын жерде сақтаңыз.", False),
]


@pytest.mark.parametrize("src,tgt,should_flag", NEG_CASES)
def test_negation(src, tgt, should_flag):
    assert bool(senim.check_negation(src, tgt)) == should_flag, senim.negation_counts(src, tgt)


def test_kazakh_reasons():
    out = senim.kk_reasons(["time missing or changed: 6:00", "number not in the original: 5"])
    assert len(out) == 2


# Regression tests for bugs found in the code review (v4.1)
REVIEW_CASES = [
    # times without minutes: Kazakh writes the hour only
    ("The meeting is on Thursday, March 5, at 6 pm in room 214.",
     "Жиналыс 5 наурыз, бейсенбі күні сағат 18-де 214 бөлмеде өтеді.", False),
    ("Come at 6 pm.", "Сағат 19-да келіңіз.", True),
    # thousands separators
    ("Take 1,000 mg daily.", "Күніне 1000 мг ішіңіз.", False),
    ("Take 1,000 mg daily.", "Күніне 1 000 мг ішіңіз.", False),
    ("Take 1,000 mg daily.", "Күніне 100 мг ішіңіз.", True),
    # compound number words
    ("two thousand five hundred people", "2500 адам", False),
    ("one hundred and five days", "105 күн", False),
    # "march" the verb is not the month
    ("The soldiers march to the city.", "Сарбаздар қалаға қарай жүреді.", False),
]


@pytest.mark.parametrize("src,tgt,should_flag", REVIEW_CASES)
def test_review_fixes(src, tgt, should_flag):
    assert bool(senim.hard_checks(src, tgt)) == should_flag, senim.hard_checks(src, tgt)


def test_nouns_are_not_negations():
    assert senim.check_negation("This is your room.", "Бұл сіздің бөлмеңіз.") == []
    assert senim.check_negation("Your certificate will be checked.", "Анықтамаңыз тексеріледі.") == []
    # a real negative imperative is still counted
    assert senim.check_negation("Give the medicine.", "Дәріні бермеңіз.") != []


def test_string_input_is_one_text():
    assert senim._as_list("Hello there") == ["Hello there"]
    assert senim._as_list(["a", "b"]) == ["a", "b"]
