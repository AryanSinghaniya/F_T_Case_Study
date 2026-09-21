"""
validate.py — Code-based verdict decision.

IMPORTANT: The LLM never decides verdicts.  All logic here is deterministic.

A note VALIDATES a candidate row if ALL of:
  (a) it applies to that route (or ALL routes)
  (b) its validity window covers week_of
  (c) its direction is "increase"

Verdict:
  - At least one validated note → "No (justified)", matched_note_id = first match
  - No validated notes         → "Yes" (flagged, unexplained), matched_note_id = ""
"""
from __future__ import annotations

import pandas as pd
from src.notes import note_applies_to_route, note_window_covers_week


def validate_candidates(
    candidates: pd.DataFrame,
    structured_notes: pd.DataFrame,
    candidate_note_ids: dict[tuple, list[str]],
) -> pd.DataFrame:
    """
    For each candidate row, determine:
      - verdict      : "No (justified)" | "Yes"
      - matched_note : note_id string, or ""
      - closest_note : best non-qualifying note (for "unexplained" reason text)

    Parameters
    ----------
    candidates          : DataFrame of candidate rows (output of build_candidate_rows)
    structured_notes    : DataFrame from load_structured_notes
    candidate_note_ids  : dict mapping (route, week_of) → list of note_id strings
                          returned by the retriever for that row

    Returns
    -------
    candidates with added columns: verdict, matched_note_id, closest_note_id
    """
    verdicts     = []
    matched_ids  = []
    closest_ids  = []

    for _, row in candidates.iterrows():
        route   = row["route"]
        week_of = row["week_of"]
        key     = (route, week_of)
        note_ids = candidate_note_ids.get(key, [])

        verdict      = "Yes"
        matched_note = ""
        closest_note = ""
        closest_score = -1  # 0=route, 1=window, 2=direction — higher = closer

        for nid in note_ids:
            note_rows = structured_notes[structured_notes["note_id"] == nid]
            if note_rows.empty:
                continue
            note = note_rows.iloc[0]

            applies = note_applies_to_route(note["routes"], route)
            in_window = note_window_covers_week(
                note["start_date"], note["end_date"], week_of
            )
            is_increase = note["direction"] == "increase"

            if applies and in_window and is_increase:
                verdict      = "No (justified)"
                matched_note = nid
                break  # first qualifying note wins

            # Track closest non-qualifying note for template reasons
            # Tie-break by note_id (alphabetical) for determinism.
            score = sum([applies, in_window, is_increase])
            if score > closest_score or (score == closest_score and nid < closest_note):
                closest_score = score
                closest_note  = nid

        verdicts.append(verdict)
        matched_ids.append(matched_note)
        closest_ids.append(closest_note if verdict == "Yes" else "")

    result = candidates.copy()
    result["verdict"]         = verdicts
    result["matched_note_id"] = matched_ids
    result["closest_note_id"] = closest_ids
    return result
