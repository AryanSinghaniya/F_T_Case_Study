# Smart Shipping Cost Watcher

A clean, reproducible Python pipeline that watches weekly freight costs across 7 Indian routes, flags anomalies against historical and peer baselines, and explains them using structured context notes — with the LLM writing only the prose reason.

## Quick Start

```bash
pip install -r requirements.txt
python src/run.py --no-llm      # No LLM required; uses template reasons
python -m pytest tests/ -v      # Run all unit + adversarial tests
streamlit run dashboard.py      # Launch interactive dashboard
```

### Interactive Q&A
```bash
python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
python ask.py "which route had the biggest spike?"
python ask.py "show me all unexplained anomalies"
python ask.py "give me a summary"
```

### LLM Note Extraction (shows AI → validation pipeline)
```bash
python scripts/extract_notes_llm.py --dry-run          # no LLM needed
python scripts/extract_notes_llm.py --compare          # vs hand-curated
# With Gemini: set GEMINI_API_KEY in .env, LLM_PROVIDER=gemini
python scripts/extract_notes_llm.py
```

## How It Works

```
shipment_records.csv
       │
       ▼
[load.py] → parse dates, compute route, week_of (Monday)
       │
       ▼
[metrics.py]
  ├── compute_weekly_cost: SUM(cost) / SUM(qty × dist)  ← ratio of sums
  ├── add_own_history_baseline: trailing 8-week mean (strict, no look-ahead)
  └── add_peer_baseline: mean of other same-type routes this week
       │
       ▼
[flags.py] → flag when cost > OWN_THRESHOLD OR PEER_THRESHOLD
       │
       ▼  (candidates only)
[retrieve.py] → ChromaDB vector database (all-MiniLM-L6-v2 embeddings)
               → fallback: sentence-transformers in-memory
               → fallback: TF-IDF cosine similarity
        │
        ▼
[validate.py] → CODE decides: note must (a) apply to route, (b) cover week, (c) direction=increase
       │
       ├── "No (justified)" → [explain.py] LLM writes reason based only on validated note
       └── "Yes"            → [explain.py] deterministic template (no LLM)
       │
       ▼
[guardrails.py] → hard assertions on every justified row
       │
       ▼
output.csv
```

## Design and Trade-offs

### Cost per tonne-km: Ratio of Sums
`cost_per_tonne_km = SUM(freight_cost_inr) / SUM(quantity_tonnes × distance_km)`

This uses the **ratio of aggregated sums**, not the mean of per-shipment ratios. This is correct because it weights heavier/longer shipments appropriately. The mean of ratios would over-weight small shipments.

### Own-History Baseline: Trailing 8 Weeks
- Strictly before the current week (no look-ahead contamination).
- If fewer than 8 prior weeks exist, all available prior weeks are used (documented in `n_prior_weeks` column).
- With 0 prior weeks (first week), the own-history baseline is `NaN`; such rows can still be flagged by the peer baseline.
- The 8-week window is approximately 2 months — long enough to capture a trend, short enough to track seasonal changes.

### Peer Baseline: Mean of Route-Level Values
- For each week, we average the `cost_per_tonne_km` of all **other routes** in the same `route_type`, excluding the route itself.
- We use a **mean of route-level values** (not a grand weighted mean) so each route contributes equally regardless of volume.
- **Known limitation**: Long and Short route types each have only 2 routes, meaning the peer comparison is based on exactly 1 other route — highly noisy. Medium routes (3 routes each) are slightly better. This is documented; treat peer flags for Long/Short routes with caution.

### Threshold Choices

| Threshold | Value | Justification |
|---|---|---|
| `OWN_THRESHOLD` | 15% | Above the 99th percentile of own_pct_diff (0.1115 = 11.15%). Set to 15% to be conservative and capture genuine spikes above random week-to-week noise. |
| `PEER_THRESHOLD` | 20% | Peer distribution is wider (std≈9%). 20% sits well above the 90th percentile (10.55%) and captures meaningful peer divergence without over-flagging. |

Sensitivity analysis (see `phase1_analysis.py` output) shows 20 candidates at (15%, 20%) — manageable for human review. The sample output rows Delhi-Jaipur 2024-11-11, Ahmedabad-Mumbai 2025-01-20, and Mumbai-Pune 2025-09-15 are all correctly captured.

### Missing Weeks
- **No route-weeks are missing** in this dataset — every route has shipments in all 104 weeks.
- If a future dataset has missing weeks: no imputation is performed. The week simply has no row in the output.

### Fewer than 8 Prior Weeks
- Weeks 1–7 of a route's data use all available prior weeks (1–7 weeks respectively).
- This is noted in `n_prior_weeks` in the intermediate data.
- The README documents this choice explicitly, as required.

### How Validation Prevents Hallucination

1. **Structured notes** (`data/notes_structured.csv`) are pre-filled and marked UNVERIFIED for human review. The validation logic reads only from this file — not from free-text LLM judgement.
2. **Three hard conditions**: a note validates a row only when (a) it applies to that route, (b) its validity window covers `week_of`, and (c) its `direction = "increase"`.
3. **Guardrails** (`guardrails.py`) assert these conditions again on every justified row in the output. Any violation raises `AssertionError` and fails the pipeline loudly.
4. **LLM scope**: the LLM only writes the prose `reason` string. It cannot change `flagged` or `matched_note_id`. Verdicts are identical whether `LLM_PROVIDER=none` or a real LLM is used.

### LLM Note Extraction (AI → Validation Pipeline)

`scripts/extract_notes_llm.py` demonstrates meaningful LLM use:
- LLM reads raw `context_notes.csv` and extracts structured fields (routes, dates, direction)
- Output is validated through the same 3-condition logic before any verdict is made
- Side-by-side diff with hand-curated notes shows where LLM agrees/disagrees
- Falls back to deterministic heuristics with `--dry-run` if no LLM configured

### Vector Database (ChromaDB)

Retrieval uses ChromaDB as the primary backend (brief explicitly recommends this):
- **all-MiniLM-L6-v2** embeddings via sentence-transformers
- Persistent collection stored in `.chroma_store/`
- Cosine similarity scoring (visible in `verbose=True` mode)
- Automatic fallback to sentence-transformers in-memory, then TF-IDF

### Acknowledged Discrepancy: Mumbai-Pune 2025-09-15

The sample output marks this week as `Yes` (unexplained). My output marks it `No (justified)` with N003.

My reasoning: N003 (diesel price rise, 2025-05-05) has **no end date** — it is open-ended. Under the natural interpretation, open-ended notes apply indefinitely. My system applies N003 consistently to all 13 Mumbai-Pune flagged weeks after May 2025. See `WALKTHROUGH.md` §8 for the full defence.

## Reproducibility Check

```powershell
# Windows
.\repro_check.ps1

# Linux/Mac
bash repro_check.sh
```

Runs the pipeline 3 times and diffs the outputs. Result saved to `eval/repro_diff.txt`.
Because `LLM_PROVIDER=none` uses deterministic templates and pandas operations are deterministic on the same input, all 3 runs produce byte-identical output.

## Cost

With `LLM_PROVIDER=none` (default): **$0.00**. No LLM calls are made.

With Gemini 2.0 Flash (free tier): **$0.00** (free quota). ~4,800 input tokens and ~960 output tokens on first run; 0 on subsequent runs (cached).

See `token_log.md` for full details.

## Known Limitations

1. **Small peer groups**: 1 peer each for Long and Short routes. Peer comparison is noisy.
2. **N003 open-ended**: Diesel price rise has no stated end date; assumed to continue through dataset end. My system justifies all Mumbai-Pune spikes from May 2025 onward with N003. The sample output disagrees for 2025-09-15; see WALKTHROUGH §8 for full reasoning.
3. **N002 single-week window**: Festival week assumed to be exactly 1 week (Jan 20–26). If the festival lasted longer, some rows may be under-justified.
4. **Distance variation**: The same route shows slightly different `distance_km` values across shipments. Handled correctly by ratio-of-sums but worth noting.
5. **Eval labels are self-created**: `eval/labels.csv` was hand-labeled by me. Precision=Recall=1.0 demonstrates the harness infrastructure works, not independently verified ground truth.

## Project Structure

```
src/
  config.py      — All constants and paths
  load.py        — Data loading and dtype enforcement
  metrics.py     — Weekly cost, own-history, peer baselines
  flags.py       — Candidate flagging (OR rule) and formatting
  notes.py       — Structured notes loading and predicates
  retrieve.py    — ChromaDB vector DB + sentence-transformers + TF-IDF retrieval
  validate.py    — Code-based verdict (route, window, direction)
  explain.py     — LLM reason (justified) or template (unexplained)
  guardrails.py  — Hard assertions on every justified row
  run.py         — One-command pipeline runner
scripts/
  extract_notes_llm.py  — Auto-extract structured notes via LLM
  phase0_inspect.py     — Data profiling (development)
  phase1_analysis.py    — Threshold analysis (development)
  verify_notes.py       — Notes verification utility
tests/
  test_metrics.py  — Phase 1 unit tests (12 tests)
  test_validate.py — Phase 3 adversarial tests (8 tests)
eval/
  eval.py        — Precision/recall evaluation
  labels.csv     — Hand-labelled rows (self-created; see Known Limitations)
data/
  shipment_records.csv
  context_notes.csv
  notes_structured.csv  — UNVERIFIED, requires hand review
  sample_output_format_v2.csv
dashboard.py     — Streamlit dashboard (streamlit run dashboard.py)
output.csv       — Final pipeline output
ask.py           — NL Q&A CLI (LLM-based intent parsing, 5 question types)
token_log.md     — LLM usage and cost
WALKTHROUGH.md  — 10-minute talking points
requirements.txt
.env.example
```
