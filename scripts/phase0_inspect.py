"""
Phase 0: Data Inspection
Prints columns, dtypes, row counts, nulls, duplicates, and sample rows for all CSVs.
Checks date parsing, non-positive values, route-type consistency, peer counts, and missing weeks.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# ── helpers ──────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def section(title: str) -> None:
    print(f"\n-- {title} --")

# ── 1. Load all CSVs ─────────────────────────────────────────────────────────

banner("LOADING FILES")

shipments = pd.read_csv(DATA_DIR / "shipment_records.csv")
notes     = pd.read_csv(DATA_DIR / "context_notes.csv")
# sample_output_format has unquoted commas in the 'reason' field — parse with engine='python' and on_bad_lines='warn'
try:
    sample = pd.read_csv(DATA_DIR / "sample_output_format_v2.csv", engine="python", on_bad_lines="warn")
except Exception:
    # Fallback: read raw and print header only
    sample = None

# ── 2. Shipment records: basic profile ────────────────────────────────────────

banner("SHIPMENT RECORDS — PROFILE")

print(f"\nShape: {shipments.shape}")
print("\nDtypes:\n", shipments.dtypes)
print("\nFirst 3 rows:\n", shipments.head(3).to_string())
print("\nNull counts:\n", shipments.isnull().sum())
dup_count = shipments.duplicated().sum()
print(f"\nFull-row duplicates: {dup_count}")
dup_id    = shipments['shipment_id'].duplicated().sum()
print(f"Duplicate shipment_id values: {dup_id}")

section("Date parsing")
shipments['shipment_date'] = pd.to_datetime(shipments['shipment_date'])
print(f"Date range: {shipments['shipment_date'].min().date()} → {shipments['shipment_date'].max().date()}")
unparsed = shipments['shipment_date'].isna().sum()
print(f"Unparseable dates: {unparsed}")

section("Non-positive quantity / distance / cost")
for col in ['quantity_tonnes', 'distance_km', 'freight_cost_inr']:
    bad = (shipments[col] <= 0).sum()
    print(f"  {col} <= 0: {bad} rows")

# ── 3. Route-type consistency ─────────────────────────────────────────────────

banner("ROUTE-TYPE CONSISTENCY")

shipments['route'] = shipments['origin'] + '-' + shipments['destination']
route_types = (
    shipments.groupby('route')['route_type']
    .nunique()
    .reset_index()
    .rename(columns={'route_type': 'n_distinct_types'})
)
bad_routes = route_types[route_types['n_distinct_types'] > 1]
if bad_routes.empty:
    print("✓ Every route maps to exactly one route_type.")
else:
    print("✗ Routes with inconsistent route_type:")
    for _, row in bad_routes.iterrows():
        vals = shipments.loc[shipments['route'] == row['route'], 'route_type'].unique()
        print(f"  {row['route']}: {vals}")

# route_type per route
rt_map = shipments.groupby('route')['route_type'].first().reset_index()
print("\nRoute → route_type mapping:")
print(rt_map.to_string(index=False))

section("Routes per route_type (peer group sizes)")
peers = rt_map.groupby('route_type').size().reset_index(name='n_routes')
print(peers.to_string(index=False))

# ── 4. Weekly coverage analysis ───────────────────────────────────────────────

banner("WEEKLY COVERAGE ANALYSIS")

# Compute week_of = Monday of shipment_date
shipments['week_of'] = shipments['shipment_date'] - pd.to_timedelta(
    shipments['shipment_date'].dt.dayofweek, unit='D'
)

all_weeks  = pd.date_range(
    start=shipments['week_of'].min(),
    end=shipments['week_of'].max(),
    freq='W-MON'
)

section("Weeks without any shipment at all")
weeks_with_data = set(shipments['week_of'].unique())
missing_global  = [w for w in all_weeks if w not in weeks_with_data]
print(f"  {len(missing_global)} weeks have NO shipments at all")
if missing_global:
    for w in missing_global:
        print(f"    {w.date()}")

section("Route-weeks with no shipments (missing route × week combos)")
route_week = shipments.groupby(['route', 'week_of']).size().reset_index(name='n_shipments')
all_routes = shipments['route'].unique()
full_index = pd.MultiIndex.from_product([all_routes, all_weeks], names=['route', 'week_of'])
full_df    = pd.DataFrame(index=full_index).reset_index()
merged     = full_df.merge(route_week, on=['route', 'week_of'], how='left')
missing_rw = merged[merged['n_shipments'].isna()]
print(f"  Total missing route-week combos: {len(missing_rw)}")
print(f"  (out of {len(full_df)} possible route × week combinations)")

section("Missing weeks per route")
for route in sorted(all_routes):
    route_weeks = set(shipments.loc[shipments['route']==route, 'week_of'].unique())
    missing = [w for w in all_weeks if w not in route_weeks]
    print(f"  {route}: {len(missing)} missing weeks ({len(route_weeks)} present)")

section("Peer-group analysis: weeks where a route has NO peers (same route_type, same week)")
rt_week = shipments.groupby(['route_type', 'week_of', 'route']).size().reset_index()
rt_week_count = rt_week.groupby(['route_type', 'week_of'])['route'].nunique().reset_index(name='n_routes_present')
# peer count for a given route = n_routes_present - 1 (excluding itself)
no_peer_weeks = rt_week_count[rt_week_count['n_routes_present'] <= 1]
print(f"  Route-type × week combinations where a route has 0 peers: {len(no_peer_weeks)}")
if len(no_peer_weeks) > 0:
    print(no_peer_weeks.to_string(index=False))

# ── 5. Notes file ─────────────────────────────────────────────────────────────

banner("CONTEXT NOTES — PROFILE")

print(f"\nShape: {notes.shape}")
print("\nDtypes:\n", notes.dtypes)
print("\nAll notes:\n", notes.to_string(index=False))
print("\nNull counts:\n", notes.isnull().sum())

# ── 6. Sample output file ─────────────────────────────────────────────────────

banner("SAMPLE OUTPUT FORMAT — PROFILE")

with open(DATA_DIR / "sample_output_format_v2.csv", "r") as f:
    raw_lines = f.readlines()

header_cols = raw_lines[0].strip().split(",")
print(f"\nHeader (raw split): {header_cols}")
print(f"Number of header columns: {len(header_cols)}")
print("\nRaw lines:")
for i, line in enumerate(raw_lines, 1):
    print(f"  Line {i}: {line.rstrip()[:120]}...")  # truncate long lines

# The output contract: confirmed columns from the header
EXPECTED_OUTPUT_COLUMNS = header_cols
print(f"\nExpected output columns (exact): {EXPECTED_OUTPUT_COLUMNS}")

# ── 7. Summary of data-quality issues ─────────────────────────────────────────

banner("DATA QUALITY SUMMARY & PROPOSED HANDLING")

print("""
Issues found and proposed handling:

1. MISSING ROUTE-WEEKS
   Many route × week combinations have no shipments (routes are not served
   every single week). The brief says "report weeks in which a route has no
   shipments" — we handle this by simply having no row in the weekly aggregate
   for those combinations. No imputation or padding.

2. PEER GROUP SIZE
   Each route_type typically has 2-3 routes. This makes the peer baseline
   noisy (mean of 1-2 OTHER routes). We will document this limitation clearly.

3. N003 WINDOW
   N003 says "starting this week" (2025-05-05). Proposed interpretation:
   start_date = 2025-05-05 (the Monday of that week), end_date = open-ended
   (until superseded). WILL ASK USER before deciding.

4. NON-POSITIVE VALUES
   To be checked — if any exist they must be investigated before use.

5. DISTANCE VARIATION PER ROUTE
   The same route (e.g. Mumbai-Pune) shows slightly varying distance_km across
   shipments (likely GPS/path variation). This is expected and correct: we use
   SUM(freight_cost_inr) / SUM(quantity_tonnes * distance_km) per week, which
   naturally handles variable distances.

6. NO BRIEF.PDF FOUND IN WORKSPACE
   The file BRIEF.pdf was not present in the workspace root. We work from the
   detailed specification in the user prompt which supersedes the PDF.
""")
