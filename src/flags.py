"""
flags.py — Candidate flagging and output formatting.

A route-week is a CANDIDATE (before RAG validation) when it exceeds
EITHER the own-history threshold OR the peer threshold (OR rule).

Null own-history rows (week 1, no prior data) are still included as
candidates if they exceed the peer threshold.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.config import OWN_THRESHOLD, PEER_THRESHOLD


def flag_candidates(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'is_candidate' column.

    Flagging rule (OR):
      - own_pct_diff  > OWN_THRESHOLD  (cost above own-history mean)
      - peer_pct_diff > PEER_THRESHOLD (cost above peer mean)

    NaN comparisons are treated as False (no flag).
    """
    own_flag  = weekly["own_pct_diff"].fillna(0.0)  > OWN_THRESHOLD
    peer_flag = weekly["peer_pct_diff"].fillna(0.0) > PEER_THRESHOLD

    weekly = weekly.copy()
    weekly["is_candidate"] = own_flag | peer_flag
    return weekly


def format_pct(value: float | None, label: str) -> str:
    """
    Format a percentage difference for output.

    Example:
        format_pct(0.355, "this route's past average")
        -> "+35.5% vs this route's past average"
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"N/A vs {label}"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}% vs {label}"


def build_candidate_rows(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to candidate rows and format the vs_own_history /
    vs_similar_routes display strings.

    Returns a DataFrame ready for the RAG/explanation step,
    with columns matching the output contract.
    """
    candidates = weekly[weekly["is_candidate"]].copy()

    candidates["vs_own_history"] = candidates["own_pct_diff"].apply(
        lambda v: format_pct(v, "this route's past average")
    )
    candidates["vs_similar_routes"] = candidates["peer_pct_diff"].apply(
        lambda v: format_pct(v, "similar-length routes this week")
    )

    return candidates.reset_index(drop=True)


def percentile_report(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of key percentiles for own_pct_diff and peer_pct_diff.
    Used to justify threshold choices.
    """
    pcts = [50, 75, 80, 85, 90, 92, 95, 97, 99]
    rows = []
    for col in ("own_pct_diff", "peer_pct_diff"):
        data = weekly[col].dropna()
        # Only positive deviations are relevant for flagging
        positive = data[data > 0]
        for p in pcts:
            rows.append({
                "metric": col,
                "percentile": p,
                "all_values": round(data.quantile(p / 100), 4),
                "positive_only": round(positive.quantile(p / 100), 4) if len(positive) else float("nan"),
            })
    return pd.DataFrame(rows)
