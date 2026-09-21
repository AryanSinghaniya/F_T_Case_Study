"""
phase1_analysis.py — Run the Phase 1 pipeline and report:
  - Percentile table for own_pct_diff and peer_pct_diff
  - Recommended thresholds with justification
  - Number of candidates at those thresholds
  - Sample candidate rows
"""
import sys
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from src.load import load_shipments
from src.metrics import compute_weekly_cost, add_own_history_baseline, add_peer_baseline
from src.flags import flag_candidates, percentile_report, build_candidate_rows
from src.config import OWN_THRESHOLD, PEER_THRESHOLD

# ── Load & compute ────────────────────────────────────────────────────────────
print("Loading shipments...", flush=True)
df = load_shipments()
print(f"  {len(df)} shipments loaded.")

print("Computing weekly cost_per_tonne_km...", flush=True)
weekly = compute_weekly_cost(df)
print(f"  {len(weekly)} route-week rows.")

print("Adding own-history baseline...", flush=True)
weekly = add_own_history_baseline(weekly)

print("Adding peer baseline...", flush=True)
weekly = add_peer_baseline(weekly)

# ── Sanity: check first/last few rows ─────────────────────────────────────────
print("\n-- Sample of weekly data (first 5 rows, Mumbai-Pune) --")
sample = weekly[weekly["route"] == "Mumbai-Pune"].head(5)
print(sample[["route", "week_of", "cost_per_tonne_km",
              "own_history_mean", "n_prior_weeks",
              "peer_mean", "own_pct_diff", "peer_pct_diff"]].to_string(index=False))

# ── Percentile tables ─────────────────────────────────────────────────────────
print("\n== PERCENTILE TABLE ==")
ptable = percentile_report(weekly)
print(ptable.to_string(index=False))

# ── Distribution summary ──────────────────────────────────────────────────────
print("\n== DISTRIBUTION SUMMARY ==")
for col, label in [("own_pct_diff", "Own-history diff"), ("peer_pct_diff", "Peer diff")]:
    data = weekly[col].dropna()
    pos = data[data > 0]
    print(f"\n{label}:")
    print(f"  Total non-null rows   : {len(data)}")
    print(f"  Positive deviations   : {len(pos)} ({100*len(pos)/len(data):.1f}%)")
    print(f"  Mean (all)            : {data.mean():.4f}")
    print(f"  Std (all)             : {data.std():.4f}")
    print(f"  Min / Max             : {data.min():.4f} / {data.max():.4f}")

# ── Candidate counts at different threshold combos ────────────────────────────
print("\n== CANDIDATE COUNT SENSITIVITY ANALYSIS ==")
print(f"{'OWN':>6} {'PEER':>6} {'#Candidates':>12} {'% of all rows':>14}")
print("-" * 44)
total_rows = len(weekly)
for own_t in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    for peer_t in [0.15, 0.18, 0.20, 0.22, 0.25]:
        own_flag  = weekly["own_pct_diff"].fillna(0.0)  > own_t
        peer_flag = weekly["peer_pct_diff"].fillna(0.0) > peer_t
        n = (own_flag | peer_flag).sum()
        print(f"{own_t:>6.0%} {peer_t:>6.0%} {n:>12d} {100*n/total_rows:>13.1f}%")

# ── Apply configured thresholds ───────────────────────────────────────────────
print(f"\n== USING CONFIGURED THRESHOLDS: OWN={OWN_THRESHOLD:.0%}, PEER={PEER_THRESHOLD:.0%} ==")
weekly = flag_candidates(weekly)
candidates = build_candidate_rows(weekly)
print(f"  Candidate rows: {len(candidates)}")
print(f"  ({100*len(candidates)/total_rows:.1f}% of all route-week rows)")

# ── By-route breakdown ────────────────────────────────────────────────────────
print("\n-- Candidates by route --")
by_route = (
    candidates.groupby("route")
    .agg(n_flagged=("route", "count"),
         own_min=("own_pct_diff", lambda x: f"{100*x.min():.1f}%"),
         own_max=("own_pct_diff", lambda x: f"{100*x.max():.1f}%"),
         peer_min=("peer_pct_diff", lambda x: f"{100*x.min():.1f}%"),
         peer_max=("peer_pct_diff", lambda x: f"{100*x.max():.1f}%"))
    .reset_index()
)
print(by_route.to_string(index=False))

# ── Sample candidate rows ─────────────────────────────────────────────────────
print("\n-- Sample candidate rows (10 rows) --")
display_cols = ["route", "week_of", "cost_per_tonne_km",
                "vs_own_history", "vs_similar_routes", "n_prior_weeks"]
print(candidates[display_cols].head(10).to_string(index=False))

print("\n-- Candidates flagged by OWN history only --")
own_only = candidates[
    (candidates["own_pct_diff"].fillna(0) > OWN_THRESHOLD) &
    (candidates["peer_pct_diff"].fillna(0) <= PEER_THRESHOLD)
]
print(f"  Count: {len(own_only)}")

print("\n-- Candidates flagged by PEER only --")
peer_only = candidates[
    (candidates["own_pct_diff"].fillna(0) <= OWN_THRESHOLD) &
    (candidates["peer_pct_diff"].fillna(0) > PEER_THRESHOLD)
]
print(f"  Count: {len(peer_only)}")

print("\n-- Candidates flagged by BOTH --")
both = candidates[
    (candidates["own_pct_diff"].fillna(0) > OWN_THRESHOLD) &
    (candidates["peer_pct_diff"].fillna(0) > PEER_THRESHOLD)
]
print(f"  Count: {len(both)}")

print("\nPhase 1 analysis complete.")
