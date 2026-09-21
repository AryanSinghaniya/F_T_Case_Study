"""
metrics.py — Weekly cost-per-tonne-km computation and baselines.

Design choices documented here:
  1. cost_per_tonne_km uses the RATIO OF SUMS (not mean of per-shipment ratios)
     to correctly weight shipments by their volume and distance.
  2. Own-history baseline uses a strict trailing 8-week window (no look-ahead).
     If fewer than 8 prior weeks exist, all available prior weeks are used.
  3. Peer baseline is the mean of route-level weekly values across all OTHER
     routes with the same route_type in the same week.  We use a mean of
     route-level values (not a grand weighted mean) so each route contributes
     equally regardless of volume.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.config import HISTORY_WEEKS


# ---------------------------------------------------------------------------
# Step 1: Weekly aggregation — ratio of sums
# ---------------------------------------------------------------------------

def compute_weekly_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-shipment data to weekly cost_per_tonne_km.

    cost_per_tonne_km = SUM(freight_cost_inr) / SUM(quantity_tonnes * distance_km)

    Returns a DataFrame with columns:
      route, week_of, route_type, cost_per_tonne_km, n_shipments
    """
    df = df.copy()
    df["tkm"] = df["quantity_tonnes"] * df["distance_km"]

    agg = (
        df.groupby(["route", "week_of", "route_type"], as_index=False)
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tkm=("tkm", "sum"),
            n_shipments=("shipment_id", "count"),
        )
    )

    agg["cost_per_tonne_km"] = agg["total_cost"] / agg["total_tkm"]
    return agg[["route", "week_of", "route_type", "cost_per_tonne_km", "n_shipments"]]


# ---------------------------------------------------------------------------
# Step 2: Own-history baseline (trailing 8 weeks, strict no look-ahead)
# ---------------------------------------------------------------------------

def add_own_history_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add own-history baseline columns to the weekly DataFrame.

    For each (route, week_of):
      - own_history_mean : mean of cost_per_tonne_km over the trailing
                           HISTORY_WEEKS weeks strictly before week_of.
                           NaN if no prior weeks exist.
      - n_prior_weeks    : how many prior weeks were actually used (0 if none).
      - own_pct_diff     : (cost_per_tonne_km - own_history_mean) / own_history_mean
                           NaN when own_history_mean is NaN.
    """
    weekly = weekly.sort_values(["route", "week_of"]).copy()
    results = []

    for route, grp in weekly.groupby("route"):
        grp = grp.sort_values("week_of").reset_index(drop=True)
        means = []
        n_used = []

        for i, row in grp.iterrows():
            current_week = grp.at[i, "week_of"]
            # Select rows strictly before current_week
            prior = grp[grp["week_of"] < current_week]
            # Take the most recent HISTORY_WEEKS weeks
            window = prior.tail(HISTORY_WEEKS)
            if len(window) == 0:
                means.append(np.nan)
                n_used.append(0)
            else:
                means.append(window["cost_per_tonne_km"].mean())
                n_used.append(len(window))

        grp["own_history_mean"] = means
        grp["n_prior_weeks"] = n_used
        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    # Percentage difference vs own history
    out["own_pct_diff"] = (
        (out["cost_per_tonne_km"] - out["own_history_mean"]) / out["own_history_mean"]
    )
    return out


# ---------------------------------------------------------------------------
# Step 3: Peer baseline (same route_type, same week, other routes)
# ---------------------------------------------------------------------------

def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add peer-baseline columns to the weekly DataFrame.

    For each (route, week_of):
      - peer_mean    : mean of cost_per_tonne_km across all OTHER routes
                       with the same route_type in the same week.
                       NaN if no peers present.
      - peer_pct_diff: (cost_per_tonne_km - peer_mean) / peer_mean

    Design: peer_mean is a mean of route-level values (each route contributes
    equally), not a weighted grand mean.  This gives consistent comparison
    regardless of volume differences between peer routes.
    """
    weekly = weekly.copy()

    # Group by (route_type, week_of) to compute a sum and count that lets us
    # compute the mean of OTHER routes via exclusion.
    rt_week = (
        weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
        .agg(rt_sum="sum", rt_count="count")
        .reset_index()
    )

    merged = weekly.merge(rt_week, on=["route_type", "week_of"], how="left")

    # Peer mean = (sum of all routes in group - self) / (count - 1)
    # If count == 1 (no peers), peer_mean = NaN
    merged["peer_count_excl"] = merged["rt_count"] - 1
    merged["peer_sum_excl"] = merged["rt_sum"] - merged["cost_per_tonne_km"]
    merged["peer_mean"] = np.where(
        merged["peer_count_excl"] > 0,
        merged["peer_sum_excl"] / merged["peer_count_excl"],
        np.nan,
    )

    merged["peer_pct_diff"] = (
        (merged["cost_per_tonne_km"] - merged["peer_mean"]) / merged["peer_mean"]
    )

    return merged.drop(columns=["rt_sum", "rt_count", "peer_sum_excl", "peer_count_excl"])
