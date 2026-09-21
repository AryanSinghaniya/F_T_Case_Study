"""
ask.py — CLI for route cost queries (stretch goal).

Usage:
    python ask.py "why did Chennai-Bangalore get pricier in March 2025?"

Parses the route and date from the question, retrieves relevant notes,
validates them, and answers using only validated information.
Refuses to answer when no validated note exists.
"""
from __future__ import annotations

import sys
import re
import os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.load    import load_notes, load_notes_structured
from src.notes   import note_applies_to_route, note_window_covers_week
from src.retrieve import build_retriever, retrieve_candidates

# ── Route and date extraction (simple regex) ──────────────────────────────────

ROUTE_PATTERN = re.compile(
    r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*[-–]\s*([A-Z][a-z]+(?:-[A-Z][a-z]+)?)',
    re.IGNORECASE
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def extract_route_and_date(question: str) -> tuple[str | None, pd.Timestamp | None]:
    """Extract route (Origin-Destination) and approximate date from free text."""
    route = None
    date  = None

    # Route: look for City-City pattern
    m = ROUTE_PATTERN.search(question)
    if m:
        origin = m.group(1).strip().title()
        dest   = m.group(2).strip().title()
        route  = f"{origin}-{dest}"

    # Date: look for "March 2025" or "2025-03" patterns
    for word, month_num in MONTH_MAP.items():
        if word in question.lower():
            year_m = re.search(r'20\d{2}', question)
            year   = int(year_m.group()) if year_m else 2025
            date   = pd.Timestamp(year=year, month=month_num, day=1)
            break

    return route, date


def answer(question: str) -> str:
    """
    Answer a cost question using validated notes only.
    Returns a plain-English response or a refusal.
    """
    raw_notes        = load_notes()
    structured_notes = load_notes_structured()

    build_retriever(structured_notes, raw_notes)

    route, date = extract_route_and_date(question)
    if not route:
        return "Sorry, I could not identify a route (City-Destination) in your question."

    # Use first Monday of the identified month as the week
    week_of = date - pd.to_timedelta(date.dayofweek, unit="D") if date else None

    if week_of is None:
        return "Sorry, I could not identify a date or month in your question."

    # Retrieve candidate notes
    note_ids = retrieve_candidates(route, week_of, own_pct=None, peer_pct=None)
    note_text_map = raw_notes.set_index("note_id")["note"].to_dict()

    # Validate each
    validated_notes = []
    for nid in note_ids:
        note_rows = structured_notes[structured_notes["note_id"] == nid]
        if note_rows.empty:
            continue
        note = note_rows.iloc[0]

        if (note_applies_to_route(note["routes"], route) and
            note_window_covers_week(note["start_date"], note["end_date"], week_of) and
            note["direction"] == "increase"):
            validated_notes.append((nid, note_text_map.get(nid, "")))

    if not validated_notes:
        return (
            f"No validated explanation found for {route} around {week_of.date()}. "
            f"I can only answer based on verified context notes, and none qualify for "
            f"this route and date. This may be worth a human review."
        )

    nid, text = validated_notes[0]
    return f"[{nid}] {text}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(answer(question))
