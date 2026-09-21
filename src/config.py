"""
config.py — Central constants for the shipping cost watcher.

All thresholds and magic numbers live here so they can be changed in one place.
"""

# ---------------------------------------------------------------------------
# Thresholds for flagging
# ---------------------------------------------------------------------------
# A route-week is a CANDIDATE when it exceeds EITHER threshold (OR rule).
# See README for percentile-based justification.

OWN_THRESHOLD: float = 0.15   # +15 % above the route's own 8-week trailing mean
PEER_THRESHOLD: float = 0.20  # +20 % above the mean of peer routes this week

# ---------------------------------------------------------------------------
# Own-history window
# ---------------------------------------------------------------------------
HISTORY_WEEKS: int = 8  # trailing weeks strictly before the current week

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_PATH  = PROJECT_ROOT / "output.csv"

SHIPMENTS_CSV     = DATA_DIR / "shipment_records.csv"
NOTES_CSV         = DATA_DIR / "context_notes.csv"
NOTES_STRUCTURED  = DATA_DIR / "notes_structured.csv"
SAMPLE_OUTPUT_CSV = DATA_DIR / "sample_output_format_v2.csv"

# ---------------------------------------------------------------------------
# LLM / cache
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / ".llm_cache"

# Expected output columns in exact order
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]
