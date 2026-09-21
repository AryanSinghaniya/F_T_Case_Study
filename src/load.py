"""
load.py — Data loading and validation.

Loads shipment_records.csv and context_notes.csv, enforces dtypes,
and raises clear errors on data-quality issues.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.config import SHIPMENTS_CSV, NOTES_CSV, NOTES_STRUCTURED


def load_shipments(path: Path = SHIPMENTS_CSV) -> pd.DataFrame:
    """
    Load shipment records.

    Returns a DataFrame with:
      - shipment_date parsed to datetime
      - route column (Origin-Destination)
      - week_of column (Monday of shipment_date)
    """
    df = pd.read_csv(path)

    # --- dtypes ---
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["quantity_tonnes"] = df["quantity_tonnes"].astype(float)
    df["distance_km"] = df["distance_km"].astype(float)
    df["freight_cost_inr"] = df["freight_cost_inr"].astype(float)

    # --- basic sanity checks ---
    for col in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        bad = (df[col] <= 0).sum()
        if bad:
            raise ValueError(f"load_shipments: {bad} non-positive values in '{col}'")

    # --- derived columns ---
    df["route"] = df["origin"] + "-" + df["destination"]

    # week_of = Monday of the shipment_date (computed BEFORE aggregation, per brief)
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(
        df["shipment_date"].dt.dayofweek, unit="D"
    )
    # Normalize to date-only timestamp at midnight
    df["week_of"] = df["week_of"].dt.normalize()

    return df


def load_notes(path: Path = NOTES_CSV) -> pd.DataFrame:
    """Load raw context notes."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


