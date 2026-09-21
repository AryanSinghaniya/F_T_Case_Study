"""
eval/eval.py — Precision/recall evaluation against hand-labelled rows.

Usage:
    python eval/eval.py

Reads eval/labels.csv (route, week_of, expected_verdict, expected_note_id)
and output.csv, then reports:
  - Precision/recall for "Yes" (flagged) verdicts
  - Note ID matching accuracy for "No (justified)" rows
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(output_path: str = "output.csv", labels_path: str = "eval/labels.csv"):
    output = pd.read_csv(output_path)
    labels = pd.read_csv(labels_path)

    # Merge on route + week_of
    merged = labels.merge(
        output[["route", "week_of", "flagged", "matched_note_id"]],
        on=["route", "week_of"],
        how="left",
    )

    # For rows not in output at all, flag as "not flagged" (i.e. predict No)
    merged["flagged"] = merged["flagged"].fillna("No (not flagged)")
    merged["matched_note_id"] = merged["matched_note_id"].fillna("")

    total = len(merged)

    # ── Verdict evaluation ────────────────────────────────────────────────────
    # Positive class = "Yes" (flagged as unexpected)
    tp = ((merged["expected_verdict"] == "Yes") & (merged["flagged"] == "Yes")).sum()
    fp = ((merged["expected_verdict"] != "Yes") & (merged["flagged"] == "Yes")).sum()
    fn = ((merged["expected_verdict"] == "Yes") & (merged["flagged"] != "Yes")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    print(f"\n== VERDICT EVALUATION (positive class = 'Yes' / unexplained) ==")
    print(f"  True Positives  : {tp}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print(f"  Precision       : {precision:.3f}")
    print(f"  Recall          : {recall:.3f}")
    print(f"  F1              : {f1:.3f}")

    # ── Note ID accuracy (for justified rows) ─────────────────────────────────
    justified_labels = merged[merged["expected_verdict"] == "No (justified)"]
    if len(justified_labels) > 0:
        # Clean comparison: strip whitespace, handle blank expected as "any match is fine"
        correct_note = (
            justified_labels["expected_note_id"].fillna("").str.strip()
            == justified_labels["matched_note_id"].str.strip()
        ).sum()
        note_acc = correct_note / len(justified_labels)
        print(f"\n== NOTE ID ACCURACY (justified rows only) ==")
        print(f"  Rows: {len(justified_labels)}")
        print(f"  Correct note: {correct_note}")
        print(f"  Accuracy: {note_acc:.3f}")
    else:
        print("\nNo 'No (justified)' rows in labels to evaluate note accuracy.")

    # ── Per-row detail ────────────────────────────────────────────────────────
    print(f"\n== PER-ROW DETAIL ==")
    print(merged[["route", "week_of", "expected_verdict", "flagged",
                  "expected_note_id", "matched_note_id"]].to_string(index=False))

    return merged


if __name__ == "__main__":
    evaluate()
