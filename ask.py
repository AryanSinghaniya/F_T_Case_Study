"""
ask.py — Natural Language CLI for route cost queries (stretch goal).

Usage:
    python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
    python ask.py "which route had the biggest spike?"
    python ask.py "show me all unexplained anomalies"
    python ask.py "what happened with costs from Ahmedabad last January?"

If LLM_PROVIDER is set (.env), uses LLM to parse the question intent.
Falls back to regex parsing if no LLM is configured.

Supported question types:
  - route_date  : "why did X-Y cost more in March?"
  - biggest_spike: "which route had the biggest spike?"
  - all_unexplained: "show me all unexplained / flagged anomalies"
  - all_justified: "show me all justified / explained rows"
  - summary: "give me a summary" / "how many were flagged?"
"""
from __future__ import annotations

import sys
import re
import os
import json
import textwrap
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.load    import load_notes
from src.notes   import load_structured_notes, note_applies_to_route, note_window_covers_week
from src.retrieve import build_retriever, retrieve_candidates

OUTPUT_CSV = Path("output.csv")

# ── Known routes (for regex fallback) ────────────────────────────────────────
KNOWN_ROUTES = [
    "Mumbai-Pune", "Delhi-Jaipur", "Ahmedabad-Mumbai",
    "Chennai-Bangalore", "Mumbai-Delhi", "Kolkata-Bhubaneswar",
    "Delhi-Mumbai",
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── LLM-based intent extraction ───────────────────────────────────────────────

def _build_intent_prompt(question: str) -> str:
    return textwrap.dedent(f"""
        You are a question-understanding assistant for a freight cost analysis tool.
        Given a user question, extract a structured intent.

        Known routes: {', '.join(KNOWN_ROUTES)}

        User question: "{question}"

        Extract the intent as a JSON object with these fields:
        - "question_type": one of:
            "route_date"       — asks about a specific route and time period
            "biggest_spike"    — asks which route/week had the largest spike
            "all_unexplained"  — asks for all unexplained/flagged anomalies
            "all_justified"    — asks for all explained/justified cost spikes
            "summary"          — asks for overall counts or summary
            "unknown"          — cannot determine intent
        - "route": the route (e.g. "Chennai-Bangalore"), or null if not specified
        - "month": the month number (1-12), or null if not specified
        - "year": the year (e.g. 2025), or null if not specified

        Return ONLY the JSON, no explanation:
        {{
          "question_type": "...",
          "route": null or "City-City",
          "month": null or 1-12,
          "year": null or 2024/2025
        }}
    """).strip()


def _parse_intent_llm(question: str, provider: str) -> dict:
    """Use LLM to extract structured intent from question."""
    prompt = _build_intent_prompt(question)

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0),
        )
        raw = resp.text.strip()

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        chat = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=42,
        )
        raw = chat.choices[0].message.content.strip()

    elif provider == "ollama":
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                  "prompt": prompt,
                  "options": {"temperature": 0, "seed": 42},
                  "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
    else:
        return _parse_intent_regex(question)

    # Strip markdown fences if LLM added them
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        return _parse_intent_regex(question)


def _parse_intent_regex(question: str) -> dict:
    """Regex-based intent parsing (fallback)."""
    q = question.lower()

    # Detect question type
    if any(w in q for w in ["biggest spike", "largest spike", "most expensive", "highest cost", "which route"]):
        return {"question_type": "biggest_spike", "route": None, "month": None, "year": None}

    if any(w in q for w in ["unexplained", "flagged", "anomal", "mystery", "not justified"]):
        return {"question_type": "all_unexplained", "route": None, "month": None, "year": None}

    if any(w in q for w in ["justified", "explained", "excused", "valid reason"]):
        return {"question_type": "all_justified", "route": None, "month": None, "year": None}

    if any(w in q for w in ["summary", "how many", "total", "count", "overall"]):
        return {"question_type": "summary", "route": None, "month": None, "year": None}

    # Route detection
    route = None
    # Try "City-City" pattern
    m = re.search(r'([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s*[-–]\s*([A-Z][a-z]+(?:-[A-Z][a-z]+)?)', question, re.IGNORECASE)
    if m:
        origin = m.group(1).strip().title()
        dest   = m.group(2).strip().title()
        route  = f"{origin}-{dest}"
    else:
        # Try matching city name from known routes
        for r in KNOWN_ROUTES:
            cities = r.lower().split("-")
            if any(c in q for c in cities):
                route = r
                break

    # Date detection
    month = None
    year  = None
    for word, month_num in MONTH_MAP.items():
        if word in q:
            month = month_num
            break
    year_m = re.search(r'20(24|25)', question)
    if year_m:
        year = int(year_m.group())

    if route or month:
        return {"question_type": "route_date", "route": route, "month": month, "year": year}

    return {"question_type": "unknown", "route": None, "month": None, "year": None}


# ── Answer generators ─────────────────────────────────────────────────────────

def _answer_route_date(output_df: pd.DataFrame, raw_notes: pd.DataFrame,
                        structured_notes: pd.DataFrame, intent: dict) -> str:
    """Answer a question about a specific route/date."""
    route = intent.get("route")
    month = intent.get("month")
    year  = intent.get("year") or 2025

    if not route:
        return "Sorry, could not identify a route in your question. Try e.g. 'why did Chennai-Bangalore cost more in March?'"

    # Filter output for this route
    route_rows = output_df[output_df["route"].str.lower() == route.lower()]
    if route_rows.empty:
        return f"No flagged weeks found for route '{route}'."

    # Further filter by month/year if specified
    if month:
        route_rows["_month"] = pd.to_datetime(route_rows["week_of"]).dt.month
        route_rows["_year"]  = pd.to_datetime(route_rows["week_of"]).dt.year
        filtered = route_rows[route_rows["_month"] == month]
        if year:
            filtered = filtered[filtered["_year"] == year]
        if not filtered.empty:
            route_rows = filtered

    lines = [f"📊 Found {len(route_rows)} flagged week(s) for {route}:\n"]
    for _, r in route_rows.iterrows():
        verdict_icon = "⚠️" if r["flagged"] == "Yes" else "✅"
        lines.append(
            f"{verdict_icon} Week {r['week_of']} — {r['cost_per_tonne_km']} INR/tonne-km\n"
            f"   {r['vs_own_history']} | {r['vs_similar_routes']}\n"
            f"   Verdict: {r['flagged']}"
            + (f" (Note {r['matched_note_id']})" if pd.notna(r.get("matched_note_id")) and r.get("matched_note_id") else "")
            + f"\n   Reason: {r['reason']}\n"
        )

    return "\n".join(lines)


def _answer_biggest_spike(output_df: pd.DataFrame) -> str:
    """Find the route-week with the highest % above own history."""
    df = output_df.copy()
    # Parse pct from string like "+35.5% vs this route's past average"
    df["_own_pct"] = df["vs_own_history"].str.extract(r"([+-]?\d+\.?\d*)").astype(float)
    worst = df.loc[df["_own_pct"].idxmax()]
    return (
        f"🚨 Biggest spike: **{worst['route']}** on week **{worst['week_of']}**\n"
        f"   Cost: {worst['cost_per_tonne_km']} INR/tonne-km\n"
        f"   {worst['vs_own_history']} | {worst['vs_similar_routes']}\n"
        f"   Verdict: {worst['flagged']}\n"
        f"   Reason: {worst['reason']}"
    )


def _answer_all_unexplained(output_df: pd.DataFrame) -> str:
    """List all unexplained (Yes) anomalies."""
    rows = output_df[output_df["flagged"] == "Yes"]
    if rows.empty:
        return "No unexplained anomalies in the current output."
    lines = [f"⚠️  {len(rows)} unexplained anomalie(s):\n"]
    for _, r in rows.iterrows():
        lines.append(f"  • {r['route']} | Week {r['week_of']} | {r['vs_own_history']} — {r['reason']}")
    return "\n".join(lines)


def _answer_all_justified(output_df: pd.DataFrame) -> str:
    """List all justified (No justified) rows."""
    rows = output_df[output_df["flagged"] == "No (justified)"]
    if rows.empty:
        return "No justified rows in the current output."
    lines = [f"✅ {len(rows)} justified cost spike(s):\n"]
    for _, r in rows.iterrows():
        lines.append(f"  • {r['route']} | Week {r['week_of']} | Note {r.get('matched_note_id','')} — {r['vs_own_history']}")
    return "\n".join(lines)


def _answer_summary(output_df: pd.DataFrame) -> str:
    """Provide an overall summary."""
    total      = len(output_df)
    yes_count  = (output_df["flagged"] == "Yes").sum()
    just_count = (output_df["flagged"] == "No (justified)").sum()
    return (
        f"📈 Summary of flagged weeks:\n"
        f"  Total flagged: {total}\n"
        f"  ✅ Justified (explained): {just_count}\n"
        f"  ⚠️  Unexplained (still flagged): {yes_count}\n\n"
        f"Top unexplained routes:\n" +
        "\n".join(
            f"  • {r['route']} ({r['week_of']})"
            for _, r in output_df[output_df["flagged"] == "Yes"].iterrows()
        )
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def answer(question: str) -> str:
    """
    Answer a natural-language question about shipping costs.
    Uses LLM for intent parsing when available, regex fallback otherwise.
    """
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "none").lower()

    # Extract intent
    if provider not in ("none", ""):
        try:
            intent = _parse_intent_llm(question, provider)
        except Exception:
            intent = _parse_intent_regex(question)
    else:
        intent = _parse_intent_regex(question)

    q_type = intent.get("question_type", "unknown")

    # Load data
    raw_notes        = load_notes()
    structured_notes = load_structured_notes()

    # Load output if exists
    output_df = None
    if OUTPUT_CSV.exists():
        output_df = pd.read_csv(OUTPUT_CSV)

    # Answer by type
    if q_type == "biggest_spike" and output_df is not None:
        return _answer_biggest_spike(output_df)

    elif q_type == "all_unexplained" and output_df is not None:
        return _answer_all_unexplained(output_df)

    elif q_type == "all_justified" and output_df is not None:
        return _answer_all_justified(output_df)

    elif q_type == "summary" and output_df is not None:
        return _answer_summary(output_df)

    elif q_type == "route_date":
        if output_df is not None and intent.get("route"):
            build_retriever(structured_notes, raw_notes)
            return _answer_route_date(output_df, raw_notes, structured_notes, intent)

        # Fall back to note retrieval if no output
        route = intent.get("route")
        month = intent.get("month")
        year  = intent.get("year") or 2025
        if not route:
            return "Could not identify a route. Try: 'why did Chennai-Bangalore cost more in March 2025?'"

        week_of = pd.Timestamp(year=year, month=month or 1, day=1) if month else pd.Timestamp(year=year, month=1, day=1)
        build_retriever(structured_notes, raw_notes)
        note_ids     = retrieve_candidates(route, week_of, own_pct=None, peer_pct=None)
        note_text_map = raw_notes.set_index("note_id")["note"].to_dict()

        validated = []
        for nid in note_ids:
            note_rows = structured_notes[structured_notes["note_id"] == nid]
            if note_rows.empty:
                continue
            note = note_rows.iloc[0]
            if (note_applies_to_route(note["routes"], route) and
                    note_window_covers_week(note["start_date"], note["end_date"], week_of) and
                    note["direction"] == "increase"):
                validated.append((nid, note_text_map.get(nid, "")))

        if not validated:
            return (
                f"No validated explanation found for {route} around {week_of.date()}. "
                f"I can only answer based on verified context notes. "
                f"This may be worth a human review."
            )
        nid, text = validated[0]
        return f"[{nid}] {text}"

    else:
        return (
            "I can answer these types of questions:\n"
            "  • 'Why did Chennai-Bangalore cost more in March 2025?'\n"
            "  • 'Which route had the biggest spike?'\n"
            "  • 'Show me all unexplained anomalies'\n"
            "  • 'Show me all justified cost spikes'\n"
            "  • 'Give me a summary'\n\n"
            f"(Could not understand: '{question}')"
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        print("\nExamples:")
        print("  python ask.py \"why did Chennai-Bangalore get pricier in March 2025?\"")
        print("  python ask.py \"which route had the biggest spike?\"")
        print("  python ask.py \"show me all unexplained anomalies\"")
        print("  python ask.py \"what happened with costs from Ahmedabad last January?\"")
        print("  python ask.py \"give me a summary\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(answer(question))
