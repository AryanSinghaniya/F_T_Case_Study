# 🚛 FreightTiger — Smart Shipping Cost Watcher & Anomaly Detection System

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://freight-cost-watcher.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end, production-ready Freight Cost Watcher and Anomaly Detection engine that analyzes multi-lane freight shipments across Indian routes, detects cost spikes against historical & peer baselines, and cross-references qualitative contextual notes using **ChromaDB Vector Retrieval** and **Groq LLM** to classify spikes as **Justified** or **Unexplained**.

---

## 🔗 Live Links
- **🌐 Live Deployed Web Dashboard**: [https://freight-cost-watcher.streamlit.app/](https://freight-cost-watcher.streamlit.app/)
- **📁 GitHub Repository**: [https://github.com/AryanSinghaniya/F_T_Case_Study](https://github.com/AryanSinghaniya/F_T_Case_Study)

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AryanSinghaniya/F_T_Case_Study.git
cd F_T_Case_Study
pip install -r requirements.txt
```

### 2. Configure LLM (Optional — Free Groq / Gemini)
Copy `.env.example` to `.env` and add your free Groq API key:
```bash
cp .env.example .env
```
In `.env`:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```
*(If no API key is provided, the pipeline automatically falls back to deterministic template-based reasons — verdicts remain 100% identical!)*

### 3. Run Pipeline, Tests & Dashboard
```bash
# Run the full anomaly detection pipeline
python src/run.py

# Run all 20 unit and adversarial test suites
pytest tests/ -v

# Launch the interactive Streamlit Web Dashboard locally
streamlit run dashboard.py
```

### 4. Interactive Natural Language Q&A (CLI)
```bash
python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
python ask.py "which route had the biggest spike?"
python ask.py "show me all unexplained anomalies"
python ask.py "give me an overall summary"
```

---

## 🏗️ System Architecture & Data Flow

```
shipment_records.csv (2,940 shipments, 7 routes, 104 weeks)
       │
       ▼
[src/load.py] ─── Parse ISO dates, compute route keys, align to Monday week_of
       │
       ▼
[src/metrics.py]
  ├── compute_weekly_cost: SUM(cost) / SUM(qty × dist)  [Weighted Ratio of Sums]
  ├── add_own_history_baseline: Trailing 8-week lookback mean (Strictly prior, no look-ahead)
  └── add_peer_baseline: Mean of peer routes in same category (Short/Medium/Long)
       │
       ▼
[src/flags.py] ── Flag route-weeks exceeding OWN_THRESHOLD (15%) OR PEER_THRESHOLD (20%)
       │         (Produces 20 candidate rows = 2.7% of total volume)
       ▼
[src/retrieve.py] ── ChromaDB Vector Database (all-MiniLM-L6-v2 embeddings)
       │             Fallback hierarchy: ChromaDB → Sentence-Transformers → TF-IDF
       ▼
[src/validate.py] ── Deterministic Code Verdict Engine (Zero Hallucination):
       │             1. Route Match (Exact route string check)
       │             2. Temporal Window Match (start_date <= week_of <= end_date)
       │             3. Impact Direction Match (direction == "increase")
       ├── "No (justified)" ──► [src/explain.py] Groq LLM generates plain-English prose
       └── "Yes"            ──► [src/explain.py] Deterministic audit template
       │
       ▼
[src/guardrails.py] ── Hard assertions verifying 100% integrity of all verdicts
       │
       ▼
output.csv (20 flagged rows, fully classified and explained)
```

---

## 🎯 Key Design Decisions & Methodology

### 1. Cost per Tonne-Km: Ratio of Sums
$$\text{Cost per Tonne-Km} = \frac{\sum \text{Freight Cost (INR)}}{\sum (\text{Quantity Tonnes} \times \text{Distance Km})}$$
- Aggregates sums before division rather than averaging individual ratios. This accurately weights high-capacity shipments and prevents small consignments from distorting lane economics.

### 2. Historical Baseline: Strict 8-Week Lookback
- Uses an 8-week trailing window strictly prior to the current week to avoid any look-ahead contamination.
- Early weeks (weeks 1–7) dynamically utilize all available history ($N < 8$) and record `n_prior_weeks`.

### 3. Dual-Baseline Threshold Justification
- **Own History (+15%)**: Sits above the 99th percentile of historical variance (+11.15%), capturing genuine spikes.
- **Peer Route (+20%)**: Sits above the 90th percentile of peer group deviation (+10.55%), filtering cross-lane divergences.

### 4. Zero Hallucination Guarantee: Code-Governed Verdicts
- **Verdicts are NEVER decided by the LLM.**
- Route matching, date window boundaries, and direction filtering are executed strictly in Python code (`src/validate.py`).
- The LLM is strictly constrained to authoring human-readable explanations using only validated fact fragments.

---

## ⚖️ Discrepancy Analysis: Mumbai-Pune (2025-09-15)

In the sample output, `Mumbai-Pune` on `2025-09-15` was categorized as `Yes (Unexplained)` referencing `N006`. However, this system classifies it as **`No (Justified)` with `N003`**.

### Architectural Defense:
1. **Note N003** (*"Diesel prices rose nationwide starting this week"*, effective 2025-05-05) has **no end date (open-ended)**.
2. Under standard operational interpretation, fuel price increases persist until explicitly rescinded.
3. Therefore, $2025\text{-}05\text{-}05 \le 2025\text{-}09\text{-}15$, making `N003` a valid and direct justification for elevated freight rates.
4. Note `N006` explicitly states "stable demand" (`direction = "not_applicable"`), making it disqualified under cost-increase validation rules.
5. Our system maintains **100% internal consistency**: all 13 Mumbai-Pune flagged weeks post-May 2025 are systematically justified by `N003`.

---

## 📊 Live Web Dashboard Features

The Streamlit dashboard (`dashboard.py` / [Live URL](https://freight-cost-watcher.streamlit.app/)) provides:
1. **Executive KPI Scorecards**: Total Flagged Weeks, Unexplained Spikes, Justified % Coverage, and Monitored Routes.
2. **Interactive Cost Timeline**: Historical weekly trends per lane with color-coded anomaly markers.
3. **Audit Ledger (Data Table)**: Status badges (🔴 Action Needed vs 🟢 Justified) with full drilldown reasons.
4. **Context Notes Explorer**: Expandable knowledge base of all qualitative operational notes.
5. **💬 Ask Freight AI Assistant**: Built-in Groq LLM chat interface allowing operators to ask natural language questions directly in the browser.

---

## 🧪 Testing & Reproducibility

### Run Test Suite
```bash
pytest tests/ -v
```
Includes 20 tests covering:
- Mathematical correctness of weighted ratio of sums
- Baseline calculation & lookback isolation
- Multi-lane threshold flagging
- Adversarial date boundary and route mismatch tests
- Guardrail assertion enforcement

### Automated Reproducibility Verification
```powershell
# Windows
.\repro_check.ps1

# Linux / MacOS
bash repro_check.sh
```
Runs the entire pipeline 3 times consecutively and performs a strict byte-level `diff`. Result: **100% Byte-Identical Deterministic Output**.

---

## 📁 Repository Structure

```
├── .env.example                  # Template configuration for Groq / Gemini LLMs
├── .gitignore                    # Secrets & local cache exclusions
├── packages.txt                  # Linux system packages for Streamlit Cloud
├── requirements.txt              # Production Python dependencies
├── dashboard.py                  # Interactive Streamlit Web Application
├── ask.py                        # Natural Language CLI Q&A Interface
├── output.csv                    # Final pipeline output (20 flagged route-weeks)
├── demo.py                       # One-click comprehensive interview demonstration
├── WALKTHROUGH.md                # 10-minute technical talking points & defense
├── token_log.md                  # LLM execution metrics, token counts & caching logs
├── repro_check.ps1 / .sh         # Automated byte-level reproducibility test scripts
│
├── src/                          # Core Anomaly Engine
│   ├── config.py                 # Central constants, thresholds, and paths
│   ├── load.py                   # Data ingestion and schema validation
│   ├── metrics.py                # Weekly cost per tonne-km & baseline engines
│   ├── flags.py                  # Multi-baseline candidate thresholding
│   ├── notes.py                  # Structured context notes parser
│   ├── retrieve.py               # ChromaDB vector store + embedding search
│   ├── validate.py               # Deterministic 3-condition validation logic
│   ├── explain.py                # Groq LLM reason generator with SHA-256 caching
│   ├── guardrails.py             # Pre-output assertion guardrails
│   └── run.py                    # Main pipeline orchestrator
│
├── scripts/                      # Ancillary Utilities
│   ├── extract_notes_llm.py      # LLM automated note structuring pipeline
│   ├── phase0_inspect.py         # Initial exploratory data analysis
│   └── phase1_analysis.py        # Threshold sensitivity & percentile analysis
│
├── tests/                        # Comprehensive Unit & Adversarial Tests
│   ├── test_metrics.py           # Metric calculation unit tests
│   └── test_validate.py          # Validation engine & edge-case tests
│
├── eval/                         # Ground Truth Evaluation
│   ├── eval.py                   # Precision / Recall evaluation harness
│   └── labels.csv                # Annotated evaluation benchmark
│
└── data/                         # Datasets
    ├── shipment_records.csv      # Raw operational shipment logs (2,940 rows)
    ├── context_notes.csv         # Raw unstructured operational context notes
    └── notes_structured.csv      # Formatted reference notes
```

---

## 👨‍💻 Author
**Aryan Singhaniya**  
- GitHub: [@AryanSinghaniya](https://github.com/AryanSinghaniya)  
- Live App: [freight-cost-watcher.streamlit.app](https://freight-cost-watcher.streamlit.app/)
