"""
verify_notes.py — Side-by-side verification of notes_structured.csv vs context_notes.csv.

Shows each structured note next to its raw text so you can manually confirm:
  1. The routes field is correct
  2. The date window is correct
  3. The direction label is correct

Run: python verify_notes.py
"""
import pandas as pd

raw   = pd.read_csv("data/context_notes.csv")
struc = pd.read_csv("data/notes_structured.csv")
struc["end_date"] = struc["end_date"].fillna("OPEN-ENDED")

# What direction SHOULD the note have, based on keywords?
INCREASE_WORDS     = ["flooding", "surcharge", "disrupted", "detour", "diesel prices rose",
                      "pushing up", "higher trip costs"]
NEUTRAL_WORDS      = ["stable", "normal", "no significant", "absorbed", "not significantly",
                      "returned to normal", "not part of", "not affected"]
IMPROVEMENT_WORDS  = ["improved", "resurfacing", "normalized", "returned to normal"]

print("=" * 90)
print("  NOTES STRUCTURED — VERIFICATION REPORT")
print("  Check each row: routes, window, direction vs raw note text")
print("=" * 90)

all_ok = True

for _, s in struc.iterrows():
    nid = s["note_id"]
    raw_row = raw[raw["note_id"] == nid]
    if raw_row.empty:
        print(f"\n[{nid}] ERROR: not found in context_notes.csv!")
        all_ok = False
        continue

    note_text = raw_row.iloc[0]["note"]
    note_lower = note_text.lower()

    window = f"{s['start_date']} → {s['end_date']}"
    print(f"\n{'─'*90}")
    print(f"  {nid}  | routes: {s['routes']:<25} | window: {window:<30} | direction: {s['direction']}")
    print(f"  Raw: \"{note_text[:110]}...\"" if len(note_text) > 110 else f"  Raw: \"{note_text}\"")

    # Auto-flag potential mismatches
    warnings = []

    if s["direction"] == "increase":
        if not any(w in note_lower for w in INCREASE_WORDS):
            warnings.append("  !! WARN: direction=increase but no cost-rise keywords found in note text")

    if s["direction"] in ("none", "not_applicable"):
        if any(w in note_lower for w in INCREASE_WORDS):
            warnings.append(f"  !! WARN: direction={s['direction']} but cost-rise keywords found in note")

    if s["direction"] == "decrease":
        if not any(w in note_lower for w in IMPROVEMENT_WORDS):
            warnings.append("  !! WARN: direction=decrease but no improvement keywords found")

    # N004 special check: routes not in dataset
    if nid == "N004" and s["direction"] != "not_applicable":
        warnings.append("  !! WARN: N004 says routes not in dataset, should be not_applicable")

    # N003 special check: open-ended window
    if nid == "N003" and s["end_date"] != "OPEN-ENDED":
        warnings.append(f"  !! NOTE: N003 has end_date={s['end_date']}. Brief says 'starting this week' with no end. Is this intentional?")

    if warnings:
        for w in warnings:
            print(w)
        all_ok = False
    else:
        print(f"  OK")

print(f"\n{'='*90}")
if all_ok:
    print("  ALL ROWS LOOK GOOD — no automated warnings.")
else:
    print("  REVIEW WARNINGS ABOVE before trusting the pipeline output.")
print("=" * 90)

# Also show what the pipeline actually matched
print("\n\n  WHAT THE PIPELINE MATCHED (from output.csv):")
print("  " + "-"*60)
try:
    out = pd.read_csv("output.csv")
    matched = out[out["flagged"] == "No (justified)"][["route", "week_of", "matched_note_id"]].drop_duplicates("matched_note_id")
    for _, r in matched.iterrows():
        print(f"  Note {r['matched_note_id']} justified {r['route']} (e.g. week {r['week_of']})")
    unexp = out[out["flagged"] == "Yes"]
    print(f"\n  Unexplained (Yes) rows: {len(unexp)}")
    for _, r in unexp.iterrows():
        print(f"    {r['route']}  {r['week_of']}")
except Exception as e:
    print(f"  (output.csv not found: {e})")
