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

> **NEW: `scripts/extract_notes_llm.py`** demonstrates the AI → validation pipeline end-to-end:
> Raw note text → LLM extracts structured fields → validation logic checks them. This shows meaningful AI usage.

---

## 8. ⚠️ Acknowledged Discrepancy: Mumbai-Pune 2025-09-15

**The sample output marks Mumbai-Pune 2025-09-15 as `Yes` (unexplained). My output marks it as `No (justified)` with N003.**

Here is my reasoning, which I stand by:

| Factor | Detail |
|---|---|
| **Note N003** | "Diesel prices rose nationwide starting this week" (2025-05-05) |
| **N003 end date** | **Empty (open-ended)** — no end date in the raw note |
| **My interpretation** | Open-ended = applies to all future weeks from May 5, 2025 |
| **2025-09-15 check** | `2025-05-05 ≤ 2025-09-15` → True. N003 covers this week. |
| **Consistency** | ALL 13 Mumbai-Pune weeks after May 2025 are justified by N003 — same logic throughout |

**The sample output's reasoning** mentions "closest note (N006, 2025-09-22) mentions stable demand" — but N006 is `not_applicable` (no cost impact). My system correctly ignores it and finds the open-ended N003 instead.

**Bottom line:** My logic is internally consistent. If you accept that open-ended notes apply indefinitely (which is the natural reading), N003 justifies all Mumbai-Pune spikes from May 2025 onward. The sample output appears to have chosen N006 as the "closest" note by date proximity — which is a different algorithm.

---


## 8. Reproducibility (30 sec)

```powershell
.\repro_check.ps1
```

Runs 3 times, diffs all outputs. Result: **byte-identical** across all runs (deterministic templates + deterministic pandas). LLM reasons are cached by SHA-256 hash of (model, prompt, params), so they are also identical on re-runs.

---

## 9. ⚠️ Eval Labels Disclaimer

The `eval/labels.csv` ground truth was hand-created by me and scored **Precision=1.0, Recall=1.0, F1=1.0**.

> **Important caveat**: Perfect scores against your own labels is expected — the evaluation harness tests the *infrastructure* (does the pipeline produce the right format? does validation logic work?), not independently verified ground truth. A production system would require independently labeled data.

The harness is still valuable for:
- Regression testing (does a code change break any verdict?)
- Demonstrating the eval pipeline end-to-end
- Catching any future drift when thresholds or notes change

---

1. **Richer notes**: Add a temporal granularity field (day-level vs. week-level) to handle partial-week events more precisely.
2. **Better peer groups**: Cluster routes by actual distance range rather than the 3-category route_type, giving more meaningful comparisons.
3. **Confidence scores**: Report a confidence level for each flag based on how far above threshold and how many prior weeks are available.
4. **Streaming input**: Replace CSV batch loading with a streaming pipeline that can process new shipments daily.
5. **Human feedback loop**: Add a `feedback.csv` where reviewers mark false positives/negatives and tune thresholds automatically.
6. **Hallucination checker**: For LLM-generated reasons, automatically verify that every number, date, and entity in the reason text appears in the note or row data.

---

## 10. What I Would Improve (1 min)

1. **Richer notes**: Add a temporal granularity field (day-level vs. week-level) to handle partial-week events more precisely.
2. **Better peer groups**: Cluster routes by actual distance range rather than the 3-category route_type, giving more meaningful comparisons.
3. **Confidence scores**: Report a confidence level for each flag based on how far above threshold and how many prior weeks are available.
4. **Streaming input**: Replace CSV batch loading with a streaming pipeline that can process new shipments daily.
5. **Human feedback loop**: Add a `feedback.csv` where reviewers mark false positives/negatives and tune thresholds automatically.

---

## 11. Key Files (30 sec)

| File | What it does |
|---|---|
| `src/validate.py` | The heart: 3-condition deterministic verdict |
| `src/guardrails.py` | Hard assertions that can't be bypassed |
| `src/retrieve.py` | ChromaDB vector DB retrieval (sentence-transformers) |
| `data/notes_structured.csv` | **UNVERIFIED** — review before trust |
| `scripts/extract_notes_llm.py` | Auto-extract structured notes via LLM |
| `dashboard.py` | Streamlit dashboard — `streamlit run dashboard.py` |
| `ask.py` | NL Q&A CLI — handles 5 question types via LLM |
| `output.csv` | 20 flagged route-weeks, fully explained |
| `eval/labels.csv` | Ground truth for precision/recall (self-labeled) |
