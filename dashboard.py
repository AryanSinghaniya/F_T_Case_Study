"""
dashboard.py — Streamlit dashboard for the Freight Cost Watcher.

Launch:
    streamlit run dashboard.py

Deploy free:
    streamlit.io/cloud  (connect GitHub repo, set main file = dashboard.py)
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreightTiger — Cost Watcher",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_output() -> pd.DataFrame:
    path = Path("output.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["week_of"] = pd.to_datetime(df["week_of"])
    return df


@st.cache_data
def load_weekly_all() -> pd.DataFrame:
    """Load all weekly data (not just flagged) for sparklines."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.load    import load_shipments
        from src.metrics import compute_weekly_cost, add_own_history_baseline, add_peer_baseline
        from src.flags   import flag_candidates

        shipments = load_shipments()
        weekly    = compute_weekly_cost(shipments)
        weekly    = add_own_history_baseline(weekly)
        weekly    = add_peer_baseline(weekly)
        weekly    = flag_candidates(weekly)
        return weekly
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_notes_data() -> pd.DataFrame:
    try:
        return pd.read_csv(Path("data") / "context_notes.csv")
    except Exception:
        return pd.DataFrame()


# ── Load data ─────────────────────────────────────────────────────────────────
output_df = load_output()
weekly_df = load_weekly_all()
notes_df  = load_notes_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚛 FreightTiger — Smart Shipping Cost Watcher")
st.markdown(
    "Watches weekly freight costs across **7 Indian routes**, "
    "flags anomalies against historical and peer baselines, "
    "and explains them using structured context notes."
)

if output_df.empty:
    st.error("⚠️ output.csv not found. Run `python src/run.py --no-llm` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

all_routes = sorted(output_df["route"].unique().tolist())
selected_route = st.sidebar.selectbox(
    "Select Route",
    ["All Routes"] + all_routes,
    index=0,
)

verdict_filter = st.sidebar.multiselect(
    "Filter by Verdict",
    options=["Yes", "No (justified)"],
    default=["Yes", "No (justified)"],
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total      = len(output_df)
yes_count  = (output_df["flagged"] == "Yes").sum()
just_count = (output_df["flagged"] == "No (justified)").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Flagged Weeks", total)
col2.metric("⚠️ Unexplained", yes_count, delta=f"{yes_count/total*100:.0f}%", delta_color="inverse")
col3.metric("✅ Justified", just_count, delta=f"{just_count/total*100:.0f}%")
col4.metric("🛣️ Routes Monitored", len(all_routes))

st.divider()

# ── Route Cost Timeline ───────────────────────────────────────────────────────
st.subheader("📈 Route Cost Timeline")

if not weekly_df.empty:
    route_for_chart = selected_route if selected_route != "All Routes" else all_routes[0]
    if selected_route == "All Routes":
        st.info(f"Showing timeline for **{route_for_chart}**. Select a specific route in the sidebar for others.")

    route_weekly = weekly_df[weekly_df["route"] == route_for_chart].copy()
    route_weekly = route_weekly.sort_values("week_of")

    # Base chart data
    chart_data = route_weekly.set_index("week_of")[["cost_per_tonne_km"]].rename(
        columns={"cost_per_tonne_km": "Cost per tonne-km (INR)"}
    )

    # Add flagged overlay
    route_flagged = output_df[
        (output_df["route"] == route_for_chart)
    ].copy()
    route_flagged_weeks = set(route_flagged["week_of"].tolist())

    # Mark flagged points
    route_weekly["flagged_point"] = route_weekly.apply(
        lambda r: r["cost_per_tonne_km"] if r["week_of"] in route_flagged_weeks else np.nan,
        axis=1
    )

    tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])
    with tab1:
        st.line_chart(chart_data, use_container_width=True, height=300)
    with tab2:
        st.bar_chart(chart_data, use_container_width=True, height=300)

    # Add annotation table for flagged weeks on this route
    if not route_flagged.empty:
        st.caption(f"⚠️ Flagged weeks on {route_for_chart}:")
        flag_summary = route_flagged[["week_of", "cost_per_tonne_km", "flagged", "matched_note_id"]].copy()
        flag_summary["week_of"] = flag_summary["week_of"].dt.strftime("%Y-%m-%d")
        st.dataframe(flag_summary, use_container_width=True, hide_index=True)

else:
    st.info("Full timeline chart requires pipeline data. Showing only flagged weeks.")

st.divider()

# ── Flagged Rows Table ────────────────────────────────────────────────────────
st.subheader("🔍 Flagged Rows Detail")

display_df = output_df.copy()
if selected_route != "All Routes":
    display_df = display_df[display_df["route"] == selected_route]
if verdict_filter:
    display_df = display_df[display_df["flagged"].isin(verdict_filter)]

display_df["week_of_str"] = display_df["week_of"].dt.strftime("%Y-%m-%d")

# Color-code verdicts
def highlight_verdict(row):
    if row["flagged"] == "Yes":
        return ["background-color: #ff4b4b22"] * len(row)
    elif row["flagged"] == "No (justified)":
        return ["background-color: #21c35422"] * len(row)
    return [""] * len(row)

show_cols = ["route", "week_of_str", "cost_per_tonne_km", "vs_own_history",
             "vs_similar_routes", "flagged", "matched_note_id", "reason"]
show_df = display_df[show_cols].rename(columns={"week_of_str": "week_of"})

st.dataframe(
    show_df.style.apply(highlight_verdict, axis=1),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Showing {len(show_df)} of {len(output_df)} flagged rows.")

st.divider()

# ── Cost Comparison Bar Chart ─────────────────────────────────────────────────
st.subheader("📊 Average Flagged Cost per Route")

avg_cost = (
    output_df.groupby("route")["cost_per_tonne_km"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"cost_per_tonne_km": "Avg Cost (INR/tonne-km)"})
    .set_index("route")
)
st.bar_chart(avg_cost, use_container_width=True, height=300)

st.divider()

# ── Context Notes Explorer ────────────────────────────────────────────────────
st.subheader("📋 Context Notes Explorer")

if not notes_df.empty:
    for _, note in notes_df.iterrows():
        with st.expander(f"**{note['note_id']}** — {note['applies_to']} ({str(note['date'])[:10]})"):
            st.write(note["note"])
else:
    st.info("context_notes.csv not found.")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **FreightTiger Assignment** — Smart Shipping Cost Watcher  
    Pipeline: `load → metrics → flag → retrieve (ChromaDB) → validate → explain (LLM) → guardrails`  
    Run: `python src/run.py --no-llm` | Tests: `pytest -v` | Ask: `python ask.py "your question"`
    """,
    unsafe_allow_html=False,
)
