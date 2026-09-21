"""
notes.py — Load and index structured notes for retrieval and validation.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.config import NOTES_STRUCTURED


def load_structured_notes(path: Path = NOTES_STRUCTURED) -> pd.DataFrame:
    """
    Load notes_structured.csv and return a clean DataFrame.

    Columns: note_id, routes (list), start_date, end_date (NaT = open-ended),
             direction, evidence_phrase
    """
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"], errors="coerce")  # NaT = open-ended
    df["direction"]  = df["direction"].str.strip().str.lower()

    # routes: "ALL" stays as "ALL"; specific routes are stored as-is (single route string)
    df["routes"] = df["routes"].str.strip()
    return df


def note_applies_to_route(note_routes: str, route: str) -> bool:
    """
    Return True if the note applies to the given route.

    'ALL' matches any route.
    Otherwise the note_routes value must exactly equal route.
    """
    return note_routes == "ALL" or note_routes == route


def note_window_covers_week(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | float,  # NaT = open-ended
    week_of: pd.Timestamp,
) -> bool:
    """
    Return True if week_of falls within [start_date, end_date].

    end_date = NaT means open-ended (applies through dataset end).
    week_of represents the Monday of the week — we check start_date <= week_of <= end_date.
    A "near-miss" (e.g. week 2025-09-15 vs note starting 2025-09-22) correctly returns False.
    """
    if pd.isna(end_date):
        return start_date <= week_of
    return start_date <= week_of <= end_date
