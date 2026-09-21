"""
guardrails.py — Post-hoc assertions for every "No (justified)" row.

For each justified row, asserts ALL of:
  1. The cited note_id exists in structured notes
  2. The note applies to that route (or ALL routes)
  3. The note's validity window covers the week_of
  4. The note's direction is "increase"

Raises AssertionError loudly on any violation.
"""
from __future__ import annotations

import pandas as pd
from src.notes import note_applies_to_route, note_window_covers_week


def run_guardrails(output: pd.DataFrame, structured_notes: pd.DataFrame) -> None:
    """
    Check all 'No (justified)' rows in output.

    Parameters
    ----------
    output           : final output DataFrame (with verdict, matched_note_id, etc.)
    structured_notes : DataFrame from load_structured_notes

    Raises
    ------
    AssertionError with a descriptive message on any violation.
    """
    justified = output[output["flagged"] == "No (justified)"]
    note_index = structured_notes.set_index("note_id")

    for _, row in justified.iterrows():
        route   = row["route"]
        week_of = pd.Timestamp(row["week_of"])
        nid     = row["matched_note_id"]

        # 1. Note exists
        assert nid in note_index.index, (
            f"GUARDRAIL FAIL: matched_note_id '{nid}' not found in structured notes "
            f"(route={route}, week={week_of.date()})"
        )

        note = note_index.loc[nid]

        # 2. Applies to route
        assert note_applies_to_route(note["routes"], route), (
            f"GUARDRAIL FAIL: note '{nid}' applies to '{note['routes']}' "
            f"but row route is '{route}' (week={week_of.date()})"
        )

        # 3. Window covers week
        assert note_window_covers_week(
            note["start_date"], note["end_date"], week_of
        ), (
            f"GUARDRAIL FAIL: note '{nid}' window "
            f"[{note['start_date'].date()}, "
            f"{'open' if pd.isna(note['end_date']) else note['end_date'].date()}] "
            f"does not cover week {week_of.date()} (route={route})"
        )

        # 4. Direction is increase
        assert note["direction"] == "increase", (
            f"GUARDRAIL FAIL: note '{nid}' direction='{note['direction']}' "
            f"but only 'increase' can justify a cost flag "
            f"(route={route}, week={week_of.date()})"
        )

    print(f"Guardrails passed: {len(justified)} justified rows verified.")
