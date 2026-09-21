"""
scripts/extract_notes_llm.py — Auto-extract structured notes from context_notes.csv using LLM.

Usage:
    python scripts/extract_notes_llm.py                     # uses LLM from .env
    python scripts/extract_notes_llm.py --dry-run           # parse only, no LLM
    python scripts/extract_notes_llm.py --out data/notes_structured_llm.csv

What it does:
  1. Reads context_notes.csv (raw free-text notes)
  2. For each note, sends a structured extraction prompt to the LLM
  3. LLM extracts: routes, start_date, end_date, direction, evidence_phrase
  4. Writes the result to data/notes_structured_llm.csv
  5. Compares with the human-curated notes_structured.csv and shows diffs

This demonstrates the AI → validation pipeline end-to-end:
  Raw note text → LLM extraction → structured fields → validation logic
"""
from __future__ import annotations

import sys
import os
import json
import re
import argparse
import textwrap
import pandas as pd
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR       = Path("data")
RAW_NOTES_PATH = DATA_DIR / "context_notes.csv"
HAND_NOTES_PATH = DATA_DIR / "notes_structured.csv"
DEFAULT_OUT    = DATA_DIR / "notes_structured_llm.csv"

# Known routes in the dataset (LLM can reference these)
KNOWN_ROUTES = [
    "Mumbai-Pune", "Delhi-Jaipur", "Ahmedabad-Mumbai",
    "Chennai-Bangalore", "Mumbai-Delhi", "Kolkata-Bhubaneswar",
    "Delhi-Mumbai",
]


def build_extraction_prompt(note_id: str, date: str, applies_to: str, note_text: str) -> str:
    return textwrap.dedent(f"""
        You are a data extraction assistant. Extract structured fields from this logistics context note.
        Return ONLY a valid JSON object — no explanation, no markdown fences.

        Note ID: {note_id}
        Note date: {date}
        Applies to (raw field): {applies_to}
        Note text: "{note_text}"

        Known routes in dataset: {', '.join(KNOWN_ROUTES)}

        Extract the following fields:
        - "routes": The route this applies to. Use the exact route name from the known routes list,
          OR "ALL" if it applies to all routes.
          If the applies_to field says something like "All Routes", use "ALL".
        - "start_date": The start date of the event (YYYY-MM-DD). Use the note date if not explicitly stated.
        - "end_date": The end date of the event (YYYY-MM-DD), or "" if open-ended / ongoing.
        - "direction": One of: "increase" (costs went up), "decrease" (costs went down),
          "none" (costs not affected), "not_applicable" (route not in dataset or no cost impact).
        - "evidence_phrase": A short 8-15 word direct quote from the note that most clearly
          explains the cost direction.

        Return ONLY this JSON (no extra text):
        {{
          "routes": "...",
          "start_date": "YYYY-MM-DD",
          "end_date": "YYYY-MM-DD or empty string",
          "direction": "increase|decrease|none|not_applicable",
          "evidence_phrase": "..."
        }}
    """).strip()


def call_llm(prompt: str, provider: str) -> str:
    """Call the configured LLM and return the raw response string."""
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        )
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0),
        )
        return resp.text.strip()

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        chat = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama3-8b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=42,
        )
        return chat.choices[0].message.content.strip()

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
        return resp.json()["response"].strip()

    raise ValueError(f"Unknown LLM provider: {provider}")


def parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response (handles markdown fences)."""
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    return json.loads(clean)


def deterministic_fallback(note_id: str, date: str, applies_to: str, note_text: str) -> dict:
    """
    Rule-based fallback when no LLM is configured.
    Approximates what the LLM would do using simple heuristics.
    """
    text_lower = note_text.lower()

    # Route
    route = "ALL"
    for r in KNOWN_ROUTES:
        if r.lower() in text_lower or r.split("-")[0].lower() in applies_to.lower():
            route = r
            break
    if "all" in applies_to.lower():
        route = "ALL"

    # Direction
    increase_words = ["higher", "increased", "surcharge", "disrupted", "flooding", "rose", "rise", "detour"]
    decrease_words = ["improved", "resurfacing", "completed", "reduced", "lower"]
    no_impact_words = ["not significantly", "remained stable", "remained normal", "without a rate change",
                       "no major disruptions", "absorbed by"]
    not_applicable_words = ["not part of this dataset", "not part of"]

    if any(w in text_lower for w in not_applicable_words):
        direction = "not_applicable"
    elif any(w in text_lower for w in no_impact_words):
        direction = "none"
    elif any(w in text_lower for w in decrease_words):
        direction = "decrease"
    elif any(w in text_lower for w in increase_words):
        direction = "increase"
    else:
        direction = "none"

    # Date extraction from text
    import re as _re
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    end_date = ""
    # Look for "to Mar 8" or "through Mar 8" patterns
    m = _re.search(r"(?:to|through|until)\s+(\w+)\s+(\d{1,2})", text_lower)
    if m:
        mon = m.group(1)[:3]
        day = m.group(2).zfill(2)
        year = date[:4]
        if mon in months:
            end_date = f"{year}-{months[mon]}-{day}"

    # Evidence phrase: first sentence or first 60 chars
    first_sentence = note_text.split(".")[0][:80]

    return {
        "routes": route,
        "start_date": date,
        "end_date": end_date,
        "direction": direction,
        "evidence_phrase": first_sentence,
    }


def extract_all_notes(
    raw_notes: pd.DataFrame,
    provider: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Extract structured fields from all notes."""
    rows = []
    total = len(raw_notes)

    for i, row in raw_notes.iterrows():
        note_id    = row["note_id"]
        date       = str(row["date"])[:10]
        applies_to = row["applies_to"]
        note_text  = row["note"]

        print(f"[{i+1}/{total}] Extracting {note_id}...", end="", flush=True)

        if dry_run or provider == "none":
            extracted = deterministic_fallback(note_id, date, applies_to, note_text)
            print(f" (deterministic fallback)")
        else:
            try:
                prompt = build_extraction_prompt(note_id, date, applies_to, note_text)
                raw_response = call_llm(prompt, provider)
                extracted    = parse_json_response(raw_response)
                print(f" direction={extracted['direction']} routes={extracted['routes']}")
            except Exception as e:
                print(f" ERROR: {e} → falling back to heuristic")
                extracted = deterministic_fallback(note_id, date, applies_to, note_text)

        rows.append({
            "note_id":        note_id,
            "routes":         extracted.get("routes", "ALL"),
            "start_date":     extracted.get("start_date", date),
            "end_date":       extracted.get("end_date", ""),
            "direction":      extracted.get("direction", "none"),
            "evidence_phrase": extracted.get("evidence_phrase", ""),
        })

    return pd.DataFrame(rows)


def compare_with_hand_curated(llm_df: pd.DataFrame, hand_df: pd.DataFrame) -> None:
    """Print side-by-side comparison of LLM extraction vs hand-curated notes."""
    print("\n" + "="*80)
    print("COMPARISON: LLM Extracted vs Hand-Curated Notes")
    print("="*80)

    merged = llm_df.merge(
        hand_df[["note_id", "routes", "direction", "end_date"]].rename(
            columns={"routes": "h_routes", "direction": "h_direction", "end_date": "h_end_date"}
        ),
        on="note_id",
        how="left",
    )

    diffs = 0
    for _, r in merged.iterrows():
        route_match     = str(r["routes"]) == str(r["h_routes"])
        direction_match = str(r["direction"]) == str(r["h_direction"])
        end_match       = str(r["end_date"]) == str(r.get("h_end_date", ""))

        status = "✅ MATCH" if (route_match and direction_match) else "⚠️  DIFF"
        if not (route_match and direction_match):
            diffs += 1

        print(f"\n{r['note_id']} [{status}]")
        print(f"  Routes:    LLM={r['routes']!r:25s}  Hand={r['h_routes']!r}")
        print(f"  Direction: LLM={r['direction']!r:25s}  Hand={r['h_direction']!r}")
        if not end_match:
            print(f"  End date:  LLM={r['end_date']!r:25s}  Hand={r.get('h_end_date', '')!r}")

    print(f"\n{'='*80}")
    print(f"Summary: {len(llm_df) - diffs}/{len(llm_df)} notes match hand-curated on route + direction.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured notes from context_notes.csv using LLM"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use deterministic fallback (no LLM). Useful for testing.")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--compare", action="store_true",
                        help="Compare output with hand-curated notes_structured.csv")
    args = parser.parse_args()

    # Load env
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "none").lower()
    if args.dry_run:
        provider = "none"

    print(f"LLM Provider: {provider}")
    print(f"Input:  {RAW_NOTES_PATH}")
    print(f"Output: {args.out}")
    print()

    raw_notes = pd.read_csv(RAW_NOTES_PATH)
    raw_notes["date"] = pd.to_datetime(raw_notes["date"]).dt.strftime("%Y-%m-%d")

    extracted = extract_all_notes(raw_notes, provider, dry_run=args.dry_run)

    out_path = Path(args.out)
    extracted.to_csv(out_path, index=False)
    print(f"\n✅ Extracted notes written to: {out_path}")
    print(extracted.to_string(index=False))

    if args.compare and HAND_NOTES_PATH.exists():
        hand_df = pd.read_csv(HAND_NOTES_PATH)
        hand_df["end_date"] = hand_df["end_date"].fillna("").astype(str)
        compare_with_hand_curated(extracted, hand_df)


if __name__ == "__main__":
    main()
