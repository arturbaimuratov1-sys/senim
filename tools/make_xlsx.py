"""Builds readable Excel files from data/ratings.csv (one sentence per cell, wrapped text).

  data/ratings_human.xlsx          - blank sheet for the author's own ratings (no AI ratings inside)
  results/ratings_claude_view.xlsx - the LLM-judge ratings, for reading
"""
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "data" / "ratings.csv", encoding="utf-8")))
ERRORS = "number,date_time,negation,omission,wrong_word,grammar,untranslated,other"


def build(path, headers, widths, getter, human=False):
    wb = Workbook(); ws = wb.active; ws.title = "ratings"
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2A6F97")
        c.alignment = Alignment(vertical="center", wrap_text=True)
    for r in rows:
        ws.append(getter(r))
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    if human:
        dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
        dv.error = "Enter a whole number 1-5"; ws.add_data_validation(dv); dv.add(f"D2:D{len(rows) + 1}")
        dv2 = DataValidation(type="list", formula1=f'"{ERRORS}"', allow_blank=True)
        ws.add_data_validation(dv2); dv2.add(f"E2:E{len(rows) + 1}")
        for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
            for c in row:
                c.fill = PatternFill("solid", fgColor="FFF8DC")
    wb.save(path)


build(root / "data" / "ratings_human.xlsx",
      ["id", "English original", "Kazakh translation", "rating 1-5", "error type"],
      [12, 55, 55, 11, 16],
      lambda r: [r["id"], r["source"], r["translation"], None, None], human=True)
build(root / "results" / "ratings_claude_view.xlsx",
      ["id", "English original", "Kazakh translation", "rating 1-5", "error type", "rater", "note"],
      [12, 50, 50, 11, 14, 22, 45],
      lambda r: [r["id"], r["source"], r["translation"], int(r["rating_1_5"]), r["error_type"], r["rater"], r["note"]])
print("ok", len(rows))
