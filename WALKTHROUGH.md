# WALKTHROUGH.md — 10-Minute Talking Points

## 1. The Problem (1 min)

Shipping costs on Indian freight routes fluctuate week to week. Some spikes are genuine anomalies (worth investigating), others have clear external causes (flooding, fuel prices, festivals). The goal: **automatically distinguish the two**, route by route, week by week, with a plain-English explanation for each flagged case.

---

## 2. Data (1 min)

- **2,940 shipments** across **7 routes** over **104 weeks** (Jan 2024 – Dec 2025)
- Every route ships every week — no missing route-weeks
- **3 route types**: Short (2 routes), Medium (3 routes), Long (2 routes)
- **10 context notes** describing external events that may or may not explain cost spikes

Key data quality observation: peer groups are small (1–2 peers per route type), making the peer baseline noisy, especially for Long and Short routes.

---

## 3. Design: Why Verdicts Are Decided in Code (2 min)

The single most important design decision: **the LLM never decides whether a row is flagged or justified**. Here's why:

| | Code | LLM |
|---|---|---|
| Route matching | ✓ exact | ✗ may hallucinate route names |
| Date window check | ✓ deterministic | ✗ may misread dates |
| Direction check | ✓ structured field | ✗ may confuse "stable" with "increase" |
| Reproducibility | ✓ byte-identical | ✗ temperature > 0 varies |
| Auditability | ✓ one code path | ✗ black box |

The LLM is **only** given: route, week, numbers, and the validated note text. It writes one plain-English sentence. If no LLM is configured, a deterministic template generates the reason instead — and the `flagged` and `matched_note_id` columns are **identical either way**.

Guardrails (`guardrails.py`) re-assert all validation conditions on every justified row in the final output, failing loudly on any violation.

---

## 4. Core Metrics (1 min)

Three key choices:

1. **cost_per_tonne_km = SUM(cost) / SUM(qty × dist)** — ratio of sums, not mean of ratios. Correctly weights heavy/long shipments.

2. **Own-history baseline**: trailing 8 weeks strictly before current week. No look-ahead. With fewer than 8 prior weeks, all available prior weeks are used (documented in `n_prior_weeks`).

3. **Peer baseline**: mean of cost_per_tonne_km across other routes of same type this week, excluding self. Mean of route-level values — each route contributes equally regardless of volume.

---

## 5. Flagging and Thresholds (1 min)

**OR rule**: flag when either baseline is exceeded.

| Threshold | Value | Why |
|---|---|---|
| Own history | 15% | Above 99th percentile (11.15%) of own_pct_diff |
| Peer | 20% | Above 90th percentile (10.55%); wider peer distribution justifies higher bar |

Result: **20 candidate rows** (2.7% of all 728 route-weeks). The sample output rows all appear:
- Delhi-Jaipur 2024-11-11: +35.5% own, +21.0% peer ✓
- Ahmedabad-Mumbai 2025-01-20: +29.5% own, +22.5% peer ✓
- Mumbai-Pune 2025-09-15: +9.2% own, +23.6% peer ✓ (flagged by peer only)

---

## 6. Results (2 min)

| Verdict | Count | Key explanation |
|---|---|---|
| No (justified) | 16 | N003 (diesel, May–Dec): 13; N001 (flooding): 2; N002 (festival): 1 |
| Yes (unexplained) | 4 | Delhi-Jaipur Nov 2024 (×2), Chennai-Bangalore Mar 2025 (×2) |

The **4 unexplained rows** are genuinely interesting:
- **Delhi-Jaipur Nov 2024**: +27–35% above history, +19–21% above peers. No note covers this period. Real anomaly.
- **Chennai-Bangalore Mar 10 & 17**: Flooding ended Mar 8 (N001). Costs remain elevated after the event. Could be road repair overhang — worth investigation.

---

## 7. Validation Anti-Hallucination (1 min)

Three layers prevent fabrication:

1. **Structured notes file** (`notes_structured.csv`): human-editable, UNVERIFIED flag prompts review before use.
2. **Three-condition code check**: route match, date window, direction=increase. All three must hold.
3. **Guardrails**: post-hoc assertions on every justified row — if any condition fails at output time, the pipeline crashes rather than silently producing wrong results.

---

## 8. Reproducibility (30 sec)

```powershell
.\repro_check.ps1
```

Runs 3 times, diffs all outputs. Result: **byte-identical** across all runs (deterministic templates + deterministic pandas). LLM reasons are cached by SHA-256 hash of (model, prompt, params), so they are also identical on re-runs.

---

## 9. What I Would Improve (1 min)

1. **Richer notes**: Add a temporal granularity field (day-level vs. week-level) to handle partial-week events more precisely.
2. **Better peer groups**: Cluster routes by actual distance range rather than the 3-category route_type, giving more meaningful comparisons.
3. **Confidence scores**: Report a confidence level for each flag based on how far above threshold and how many prior weeks are available.
4. **Streaming input**: Replace CSV batch loading with a streaming pipeline that can process new shipments daily.
5. **Human feedback loop**: Add a `feedback.csv` where reviewers mark false positives/negatives and tune thresholds automatically.
6. **Hallucination checker**: For LLM-generated reasons, automatically verify that every number, date, and entity in the reason text appears in the note or row data.

---

## 10. Key Files (30 sec)

| File | What it does |
|---|---|
| `src/validate.py` | The heart: 3-condition deterministic verdict |
| `src/guardrails.py` | Hard assertions that can't be bypassed |
| `data/notes_structured.csv` | **UNVERIFIED** — review before trust |
| `output.csv` | 20 flagged route-weeks, fully explained |
| `eval/labels.csv` | Ground truth for precision/recall |
