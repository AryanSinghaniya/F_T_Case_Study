"""
tests/test_metrics.py — Unit tests for Phase 1 logic.

Covers:
  1. Monday week_of logic (any day → correct Monday)
  2. Ratio-of-sums cost_per_tonne_km (never mean of ratios)
  3. Strict 8-week trailing window with no look-ahead
  4. Fewer-than-8-weeks case (use all available)
  5. Peer exclusion: a route is excluded from its own peer baseline
  6. Output column header matches the exact contract
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import date

# ── helpers ──────────────────────────────────────────────────────────────────

def make_shipments(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal shipments DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(
        df["shipment_date"].dt.dayofweek, unit="D"
    )
    df["week_of"] = df["week_of"].dt.normalize()
    df["shipment_id"] = [f"S{i:04d}" for i in range(len(df))]
    # Build route from origin/destination
    df["route"] = df["origin"] + "-" + df["destination"]
    return df


# ── Test 1: Monday week_of logic ─────────────────────────────────────────────

@pytest.mark.parametrize("date_str, expected_monday", [
    ("2024-01-01", "2024-01-01"),  # Monday itself
    ("2024-01-02", "2024-01-01"),  # Tuesday → previous Monday
    ("2024-01-03", "2024-01-01"),  # Wednesday
    ("2024-01-06", "2024-01-01"),  # Saturday
    ("2024-01-07", "2024-01-01"),  # Sunday
    ("2024-01-08", "2024-01-08"),  # Next Monday
    ("2024-11-15", "2024-11-11"), # A Friday → its Monday
])
def test_week_of_is_monday(date_str: str, expected_monday: str):
    """week_of must always be the Monday of the shipment date's week."""
    df = make_shipments([{
        "shipment_date": date_str,
        "origin": "A", "destination": "B", "route_type": "Short",
        "quantity_tonnes": 10.0, "distance_km": 100.0, "freight_cost_inr": 1000,
    }])
    actual = df["week_of"].iloc[0].date()
    assert str(actual) == expected_monday, f"Got {actual} for input {date_str}"


# ── Test 2: Ratio of sums (not mean of per-shipment ratios) ──────────────────

def test_cost_per_tonne_km_ratio_of_sums():
    """
    Hand-computed example:
      Shipment A: cost=1000, qty=10, dist=100  → tkm=1000, ratio=1.0
      Shipment B: cost=400,  qty=2,  dist=100  → tkm= 200, ratio=2.0

      Ratio of sums = (1000+400) / (1000+200) = 1400/1200 ≈ 1.1667
      Mean of ratios = (1.0 + 2.0) / 2 = 1.5  ← WRONG

    The implementation must return ≈ 1.1667.
    """
    from src.metrics import compute_weekly_cost

    df = make_shipments([
        {"shipment_date": "2024-01-01", "origin": "X", "destination": "Y",
         "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 100.0,
         "freight_cost_inr": 1000},
        {"shipment_date": "2024-01-03", "origin": "X", "destination": "Y",
         "route_type": "Short", "quantity_tonnes": 2.0, "distance_km": 100.0,
         "freight_cost_inr": 400},
    ])

    result = compute_weekly_cost(df)
    assert len(result) == 1
    expected = 1400 / 1200
    assert abs(result["cost_per_tonne_km"].iloc[0] - expected) < 1e-9, (
        f"Expected {expected}, got {result['cost_per_tonne_km'].iloc[0]}"
    )


# ── Test 3: Strict 8-week window, no look-ahead ───────────────────────────────

def test_own_history_no_lookahead():
    """
    For week W9, the baseline must use ONLY weeks W1..W8 (8 prior weeks).
    Week W9 itself must not be included in its own baseline.
    """
    from src.metrics import compute_weekly_cost, add_own_history_baseline

    # 10 weeks for one route; costs are 1,2,3,...,10 (in INR per tkm)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        week_start = base + pd.Timedelta(weeks=i)
        rows.append({
            "shipment_date": str(week_start.date()),
            "origin": "A", "destination": "B", "route_type": "Short",
            "quantity_tonnes": 1.0, "distance_km": 1.0,
            "freight_cost_inr": i + 1,  # cost = 1,2,...,10
        })

    df = make_shipments(rows)
    weekly = compute_weekly_cost(df)
    weekly = add_own_history_baseline(weekly)
    weekly = weekly.sort_values("week_of").reset_index(drop=True)

    # Week 1 (i=0): 0 prior weeks → NaN
    assert np.isnan(weekly.loc[0, "own_history_mean"])
    assert weekly.loc[0, "n_prior_weeks"] == 0

    # Week 9 (i=8): trailing 8 weeks = weeks 1..8, costs 1..8, mean=4.5
    # (week 9 is index 8)
    assert weekly.loc[8, "n_prior_weeks"] == 8
    assert abs(weekly.loc[8, "own_history_mean"] - 4.5) < 1e-9

    # Week 10 (i=9): trailing 8 weeks = weeks 2..9, costs 2..9, mean=5.5
    assert weekly.loc[9, "n_prior_weeks"] == 8
    assert abs(weekly.loc[9, "own_history_mean"] - 5.5) < 1e-9


# ── Test 4: Fewer-than-8-weeks case ──────────────────────────────────────────

def test_own_history_fewer_than_8_weeks():
    """
    When fewer than 8 prior weeks exist, use all available prior weeks.
    Week 3 (i=2) should use exactly 2 prior weeks (i=0, i=1).
    """
    from src.metrics import compute_weekly_cost, add_own_history_baseline

    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(4):
        rows.append({
            "shipment_date": str((base + pd.Timedelta(weeks=i)).date()),
            "origin": "A", "destination": "B", "route_type": "Short",
            "quantity_tonnes": 1.0, "distance_km": 1.0,
            "freight_cost_inr": (i + 1) * 10,  # 10, 20, 30, 40
        })

    df = make_shipments(rows)
    weekly = compute_weekly_cost(df)
    weekly = add_own_history_baseline(weekly)
    weekly = weekly.sort_values("week_of").reset_index(drop=True)

    # Week 3 (index 2): 2 prior weeks with costs 10, 20 → mean = 15
    assert weekly.loc[2, "n_prior_weeks"] == 2
    assert abs(weekly.loc[2, "own_history_mean"] - 15.0) < 1e-9

    # Week 4 (index 3): 3 prior weeks, costs 10,20,30 → mean = 20
    assert weekly.loc[3, "n_prior_weeks"] == 3
    assert abs(weekly.loc[3, "own_history_mean"] - 20.0) < 1e-9


# ── Test 5: Peer exclusion — route excluded from own peer mean ────────────────

def test_peer_excludes_self():
    """
    Route A's peer mean must NOT include Route A itself.

    Setup: two Short routes in week W1
      Route A: cost_per_tonne_km = 2.0
      Route B: cost_per_tonne_km = 4.0

    Route A's peer mean should be 4.0 (only B), not (2+4)/2=3.0.
    Route B's peer mean should be 2.0 (only A).
    """
    from src.metrics import compute_weekly_cost, add_peer_baseline

    df = make_shipments([
        {"shipment_date": "2024-01-01", "origin": "A", "destination": "X",
         "route_type": "Short", "quantity_tonnes": 1.0, "distance_km": 1.0,
         "freight_cost_inr": 2},
        {"shipment_date": "2024-01-01", "origin": "B", "destination": "Y",
         "route_type": "Short", "quantity_tonnes": 1.0, "distance_km": 1.0,
         "freight_cost_inr": 4},
    ])

    weekly = compute_weekly_cost(df)
    weekly = add_peer_baseline(weekly)

    row_a = weekly[weekly["route"] == "A-X"].iloc[0]
    row_b = weekly[weekly["route"] == "B-Y"].iloc[0]

    assert abs(row_a["peer_mean"] - 4.0) < 1e-9, f"Route A peer_mean should be 4.0, got {row_a['peer_mean']}"
    assert abs(row_b["peer_mean"] - 2.0) < 1e-9, f"Route B peer_mean should be 2.0, got {row_b['peer_mean']}"


# ── Test 6: Output column header matches contract exactly ─────────────────────

def test_output_column_header():
    """
    The output.csv header must exactly match the sample output format.
    """
    from src.config import OUTPUT_COLUMNS

    expected = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason",
    ]
    assert OUTPUT_COLUMNS == expected, (
        f"OUTPUT_COLUMNS mismatch.\nExpected: {expected}\nGot:      {OUTPUT_COLUMNS}"
    )
