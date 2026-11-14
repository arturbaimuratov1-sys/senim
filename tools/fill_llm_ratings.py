"""Writes the LLM-judge ratings into data/ratings.csv.

Rater: Claude Opus 5.5 (Anthropic), acting as an LLM judge, reading the English source and the
Kazakh translation side by side. NOT a human rating. Scale (fixed before rating):
  5 correct and natural; 4 small grammar/style issues, meaning fully kept;
  3 meaning partly wrong or something missing; 2 serious error (wrong number/date,
  flipped negation, different meaning, big omission); 1 useless or dangerous.
The judge had not seen back-translation scores for these rows.
"""
import csv
from pathlib import Path

R = {  # id: (rating, error_type, short note)
    "flores_971": (4, "wrong_word", "calques: кондименттер, тун балық"),
    "own_10": (4, "wrong_word", "алыңыз (get) instead of қабылдаңыз for medicine"),
    "own_26": (5, "", "correct; шейшембі is colloquial Tuesday"),
    "flores_134": (2, "wrong_word", "fire 'started' became 'extinguished' (сөндірді); pm and yard lost"),
    "flores_296": (4, "grammar", "meaning kept, clumsy title"),
    "flores_663": (3, "wrong_word", "'held to the same standards' became 'have excellent standards'"),
    "own_43": (2, "date_time", "Friday became Sunday (жексенбі)"),
    "flores_213": (5, "", "correct"),
    "flores_277": (5, "", "correct"),
    "flores_936": (3, "wrong_word", "flyers -> pilots (ұшқыштар); airlines 'pay' instead of 'charge'"),
    "flores_1001": (2, "omission", "second half dropped; Easter mistranslated"),
    "own_7": (5, "", "correct"),
    "flores_693": (3, "wrong_word", "Niagara Falls -> non-word шарлауығы; 'untapped' lost"),
    "flores_310": (5, "", "correct"),
    "flores_796": (4, "wrong_word", "camping areas -> camps"),
    "own_13": (5, "", "correct"),
    "own_9": (3, "grammar", "'if you make breathing difficult' instead of 'have difficulty breathing'"),
    "own_5": (4, "grammar", "25°C төмен missing case ending"),
    "flores_596": (4, "omission", "'crucially' lost"),
    "flores_493": (2, "wrong_word", "direction reversed: clothing influenced the culture"),
    "own_18": (3, "wrong_word", "'exactly as prescribed' garbled"),
    "flores_742": (5, "", "correct, all numbers kept"),
    "own_41": (4, "untranslated", "PTA left as is; spacing before punctuation"),
    "own_32": (2, "wrong_word", "'send a doctor's note' became 'send a letter to the doctor'"),
    "own_6": (4, "wrong_word", "есте ұстаған (keep in mind) instead of 'when you remember'"),
    "own_14": (3, "omission", "output cut off after 'because it'; instruction itself kept"),
    "own_48": (4, "grammar", "болу керек -> болуы керек"),
    "own_24": (4, "grammar", "infinitive instead of imperative, informal 'сен'"),
    "own_12": (3, "wrong_word", "breastfeeding -> non-word аналайтын"),
    "flores_457": (4, "grammar", "meaning kept"),
    "own_46": (3, "wrong_word", "'at short notice' -> 'in a short time'; garbled subject"),
    "own_47": (4, "wrong_word", "'open day' -> 'school will be open'"),
    "own_33": (3, "wrong_word", "morning pickup -> 'morning admission' (қабылдау)"),
    "own_35": (5, "", "correct"),
    "own_17": (2, "wrong_word", "OTC (no prescription) lost: 'can be bought with medicines'; typo"),
    "own_28": (4, "grammar", "verb missing, meaning clear"),
    "own_36": (5, "", "correct"),
    "own_37": (5, "", "correct"),
    "flores_843": (4, "wrong_word", "rider -> driver"),
    "flores_801": (2, "omission", "second sentence dropped; 'on the house' -> 'at home'"),
    "flores_926": (4, "grammar", "meaning kept"),
    "flores_78": (4, "wrong_word", "welcomed -> congratulated"),
    "flores_684": (2, "omission", "first sentence (virtual field trips) dropped"),
    "flores_611": (2, "wrong_word", "developed countries -> developing (дамушы)"),
    "flores_321": (2, "wrong_word", "'freaky' -> 'a man' (еркек); quote meaning changed"),
    "own_19": (2, "wrong_word", "'consult a doctor' -> 'give advice to the doctor'"),
    "flores_566": (5, "", "correct"),
    "flores_347": (5, "", "correct"),
    "flores_344": (3, "wrong_word", "red cabbage -> red vegetable; 'chemistry's pH'"),
    "own_29": (3, "wrong_word", "winter holidays -> winter celebration (мереке), not school break"),
    "flores_161": (4, "grammar", "meaning mostly kept"),
    "flores_981": (3, "omission", "'a little', 'hastily', 'close enough' lost"),
    "own_50": (5, "", "correct"),
    "own_31": (3, "wrong_word", "mathematics exam -> mathematics class (сыныбы)"),
    "flores_569": (4, "wrong_word", "'broadsides' mistranslated, numbers kept"),
    "own_38": (4, "grammar", "awkward but understandable"),
    "own_49": (5, "", "correct"),
    "own_16": (5, "", "correct"),
    "own_34": (5, "", "correct"),
    "own_44": (5, "", "correct, 5:00 PM -> 17:00"),
    "flores_57": (2, "wrong_word", "debate -> storm (дауыл); garbled"),
    "own_8": (2, "wrong_word", "dizziness lost, nausea repeated twice, rash -> 'skin fracture'"),
    "flores_240": (3, "omission", "reason (badly tucked wheel) dropped"),
    "flores_395": (5, "", "correct"),
    "flores_709": (5, "", "correct"),
    "own_30": (5, "", "correct"),
    "flores_606": (4, "untranslated", "orangutans transliterated oddly"),
    "own_2": (5, "", "correct, all numbers kept"),
    "flores_558": (4, "wrong_word", "rough -> cruel (қатал)"),
    "flores_756": (3, "wrong_word", "misparsed: 'Stanley credit cards'"),
    "own_3": (4, "wrong_word", "breakfast -> ертеңгі (tomorrow's) meal"),
    "flores_745": (3, "omission", "agriculture part dropped; indigenous -> compatriots"),
    "own_23": (5, "", "correct"),
    "own_27": (5, "", "correct"),
    "own_15": (4, "wrong_word", "preferably -> 'first of all'"),
    "own_40": (4, "grammar", "'so that' became 'therefore'"),
    "own_45": (2, "negation", "'may miss the trip' -> 'may NOT miss the trip'"),
    "own_21": (5, "", "correct"),
    "own_39": (5, "", "correct"),
    "flores_541": (5, "", "correct"),
    "own_11": (3, "wrong_word", "'do not drive' lost: 'do not move heavy machines or do not work'"),
    "own_42": (5, "", "correct"),
    "flores_859": (4, "grammar", "meaning kept"),
    "own_20": (4, "wrong_word", "missed dose -> 'lost dose'"),
    "own_22": (1, "negation", "'if you notice unusual bleeding' -> 'if you do NOT have usual bleeding' (dangerous)"),
    "own_1": (5, "", "correct"),
    "flores_447": (5, "", "correct"),
    "flores_397": (5, "", "correct"),
    "own_4": (5, "", "correct"),
    "own_25": (4, "wrong_word", "'everything you take' -> 'all things that treat'"),
}

path = Path(__file__).resolve().parent.parent / "data" / "ratings.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
assert {r["id"] for r in rows} == set(R), "id mismatch"
for r in rows:
    rating, err, note = R[r["id"]]
    r["rating_1_5"], r["error_type"], r["rater"], r["note"] = rating, err, "Claude Opus 5.5 (LLM judge)", note
with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "source", "translation", "rating_1_5", "error_type", "rater", "note"])
    w.writeheader()
    w.writerows(rows)
from collections import Counter
print(len(rows), "rows;", sorted(Counter(v[0] for v in R.values()).items()))
print("bad (<=3):", sum(v[0] <= 3 for v in R.values()), "| <=2:", sum(v[0] <= 2 for v in R.values()))
