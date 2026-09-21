"""
tests/test_validate.py — Adversarial tests for the validation logic.

Tests that wrong route, wrong window, non-increase notes, out-of-scope notes,
and near-miss dates all yield "Yes" (unexplained).
"""
from __future__ import annotations

import pytest
import pandas as pd
from src.validate import validate_candidates


def make_notes(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def make_candidates(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["week_of"] = pd.to_datetime(df["week_of"])
    # Add required columns with defaults
    for col in ("cost_per_tonne_km", "own_pct_diff", "peer_pct_diff",
                "vs_own_history", "vs_similar_routes", "n_shipments", "n_prior_weeks",
                "route_type", "own_history_mean", "peer_mean", "is_candidate"):
        if col not in df.columns:
            df[col] = None
    return df


# Note set used across adversarial tests
ADVERSARIAL_NOTES = make_notes([
    {"note_id": "N001", "routes": "Chennai-Bangalore",
     "start_date": "2025-02-24", "end_date": "2025-03-08",
     "direction": "increase", "evidence_phrase": "flooding"},
    {"note_id": "N004", "routes": "ALL",
     "start_date": "2024-03-11", "end_date": "2024-03-11",
     "direction": "not_applicable", "evidence_phrase": "routes not in dataset"},
    {"note_id": "N005", "routes": "Mumbai-Delhi",
     "start_date": "2024-07-29", "end_date": "2024-08-04",
     "direction": "none", "evidence_phrase": "costs not significantly affected"},
    {"note_id": "N006", "routes": "ALL",
     "start_date": "2025-09-22", "end_date": "2025-12-31",
     "direction": "not_applicable", "evidence_phrase": "stable demand"},
])


def run_validate(route: str, week: str, note_ids: list[str]) -> str:
    """Helper: validate a single candidate row and return its verdict."""
    candidates = make_candidates([{"route": route, "week_of": week}])
    key = (route, pd.Timestamp(week))
    result = validate_candidates(candidates, ADVERSARIAL_NOTES, {key: note_ids})
    return result.iloc[0]["verdict"]


# ── Test 1: Wrong route — note applies to different route ────────────────────

def test_wrong_route_yields_yes():
    """N001 applies to Chennai-Bangalore, not Mumbai-Pune → should be Yes."""
    verdict = run_validate("Mumbai-Pune", "2025-02-24", ["N001"])
    assert verdict == "Yes", f"Expected 'Yes' for wrong route, got '{verdict}'"


# ── Test 2: Wrong time window — note window doesn't cover week ───────────────

def test_wrong_window_yields_yes():
    """N001 window is Feb 24 – Mar 8; week 2025-03-16 is after → Yes."""
    verdict = run_validate("Chennai-Bangalore", "2025-03-16", ["N001"])
    assert verdict == "Yes", f"Expected 'Yes' for wrong window, got '{verdict}'"


# ── Test 3: Note says costs were unaffected (direction=none) ─────────────────

def test_direction_none_yields_yes():
    """N005 direction='none' — cannot justify a cost flag → Yes."""
    verdict = run_validate("Mumbai-Delhi", "2024-07-29", ["N005"])
    assert verdict == "Yes", f"Expected 'Yes' for direction=none, got '{verdict}'"


# ── Test 4: Out-of-scope note (N004) ─────────────────────────────────────────

def test_out_of_scope_note_yields_yes():
    """N004 direction='not_applicable' — must yield Yes even if route/window match."""
    verdict = run_validate("Mumbai-Pune", "2024-03-11", ["N004"])
    assert verdict == "Yes", f"Expected 'Yes' for N004 (not_applicable), got '{verdict}'"


# ── Test 5: Near-miss date — note starts AFTER the week ─────────────────────

def test_near_miss_date_yields_yes():
    """N006 starts 2025-09-22; week 2025-09-15 is before → Yes."""
    verdict = run_validate("Mumbai-Pune", "2025-09-15", ["N006"])
    assert verdict == "Yes", f"Expected 'Yes' for near-miss date, got '{verdict}'"


# ── Test 6: ALL routes note with direction=not_applicable ────────────────────

def test_all_routes_not_applicable_yields_yes():
    """N006 applies to ALL but direction is not_applicable → Yes."""
    verdict = run_validate("Delhi-Jaipur", "2025-10-01", ["N006"])
    assert verdict == "Yes", f"Expected 'Yes' for not_applicable ALL note, got '{verdict}'"


# ── Test 7: Valid note correctly justifies ───────────────────────────────────

def test_valid_note_yields_justified():
    """N001 correctly applies, in-window, increase → No (justified)."""
    verdict = run_validate("Chennai-Bangalore", "2025-02-24", ["N001"])
    assert verdict == "No (justified)", f"Expected justified, got '{verdict}'"


# ── Test 8: Output column header contract ────────────────────────────────────

def test_output_header_contract():
    """The OUTPUT_COLUMNS list must exactly match the sample output format."""
    from src.config import OUTPUT_COLUMNS
    expected = [
        "route", "week_of", "cost_per_tonne_km",
        "vs_own_history", "vs_similar_routes",
        "flagged", "matched_note_id", "reason",
    ]
    assert OUTPUT_COLUMNS == expected
