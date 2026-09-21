"""
run.py — One-command pipeline runner.

Usage:
    python src/run.py [--no-llm]

Steps:
  0. Load data
  1. Compute weekly metrics
  2. Flag candidates
  3. Retrieve + validate notes
  4. Generate reasons (LLM or template)
  5. Run guardrails
  6. Write output.csv
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import pandas as pd

# Ensure project root is on path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.load       import load_shipments, load_notes, load_notes_structured
from src.metrics    import compute_weekly_cost, add_own_history_baseline, add_peer_baseline
from src.flags      import flag_candidates, build_candidate_rows
from src.notes      import load_structured_notes
from src.retrieve   import build_retriever, retrieve_candidates
from src.validate   import validate_candidates
from src.explain    import generate_reasons, get_token_log
from src.guardrails import run_guardrails
from src.config     import OUTPUT_PATH, OUTPUT_COLUMNS


def run_pipeline(no_llm: bool = False) -> pd.DataFrame:
    if no_llm:
        os.environ["LLM_PROVIDER"] = "none"

    # ── 0. Load ──────────────────────────────────────────────────────────────
    print("[1/7] Loading data...")
    shipments        = load_shipments()
    raw_notes        = load_notes()
    structured_notes = load_structured_notes()

    # ── 1. Metrics ───────────────────────────────────────────────────────────
    print("[2/7] Computing weekly cost_per_tonne_km...")
    weekly = compute_weekly_cost(shipments)

    print("[3/7] Adding baselines...")
    weekly = add_own_history_baseline(weekly)
    weekly = add_peer_baseline(weekly)

    # ── 2. Flag ──────────────────────────────────────────────────────────────
    print("[4/7] Flagging candidates...")
    weekly     = flag_candidates(weekly)
    candidates = build_candidate_rows(weekly)
    print(f"      {len(candidates)} candidate rows.")

    # ── 3. Retrieval ─────────────────────────────────────────────────────────
    print("[5/7] Retrieving notes for candidates...")
    build_retriever(structured_notes, raw_notes)

    candidate_note_ids: dict[tuple, list[str]] = {}
    for _, row in candidates.iterrows():
        key = (row["route"], row["week_of"])
        candidate_note_ids[key] = retrieve_candidates(
            route   = row["route"],
            week_of = row["week_of"],
            own_pct = row.get("own_pct_diff"),
            peer_pct = row.get("peer_pct_diff"),
        )

    # ── 4. Validate ──────────────────────────────────────────────────────────
    print("[5/7] Validating notes (code-based, no LLM)...")
    validated = validate_candidates(candidates, structured_notes, candidate_note_ids)

    # ── 5. Generate reasons ──────────────────────────────────────────────────
    print("[6/7] Generating reasons...")
    with_reasons = generate_reasons(validated, structured_notes, raw_notes)

    # ── 6. Guardrails ────────────────────────────────────────────────────────
    print("[7/7] Running guardrails...")
    # Rename for guardrails (expects 'flagged' column)
    output_df = with_reasons.rename(columns={"verdict": "flagged"})
    run_guardrails(output_df, structured_notes)

    # ── 7. Write output.csv ──────────────────────────────────────────────────
    output_df["week_of"] = output_df["week_of"].dt.strftime("%Y-%m-%d")
    output_df["cost_per_tonne_km"] = output_df["cost_per_tonne_km"].round(2)
    final = output_df[OUTPUT_COLUMNS].copy()

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"Total rows: {len(final)}")
    print(f"\nVerdict summary:")
    print(final["flagged"].value_counts().to_string())

    # Token log
    log = get_token_log()
    live   = sum(1 for r in log if r["source"] == "live")
    cached = sum(1 for r in log if r["source"] == "cache")
    print(f"\nLLM calls: {live} live, {cached} cached")

    return final


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    run_pipeline(no_llm=no_llm)
