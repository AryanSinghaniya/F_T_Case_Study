"""
explain.py — Plain-English reason generation.

The LLM ONLY writes the reason string.  It is given the validated note text
and the numbers; it must not add facts not present in the note.

For "Yes" (unexplained) rows, a deterministic template is used — no LLM needed.

LLM provider is configured via .env:
  LLM_PROVIDER = ollama | groq | gemini | none
  OLLAMA_MODEL  = llama3.2 (default)
  GROQ_API_KEY  = ...
  GEMINI_API_KEY = ...

All responses are cached on disk keyed by sha256(model + prompt + params).
"""
from __future__ import annotations

import os
import json
import hashlib
import textwrap
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import CACHE_DIR

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _cache_put(key: str, value: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.txt").write_text(value, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM call (live)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, provider: str, model: str) -> str:
    """Make a single LLM call.  temperature=0, seed=42 where supported."""
    if provider == "ollama":
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt,
                  "options": {"temperature": 0, "seed": 42},
                  "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        chat = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=42,
        )
        return chat.choices[0].message.content.strip()

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0),
        )
        return resp.text.strip()

    raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Token log
# ---------------------------------------------------------------------------

_token_log: list[dict] = []


def get_token_log() -> list[dict]:
    return _token_log


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _prompt_justified(
    route: str,
    week_of: str,
    cost_per_tkm: float,
    own_pct: str,
    peer_pct: str,
    note_id: str,
    note_text: str,
) -> str:
    return textwrap.dedent(f"""
        You are writing one plain-English sentence explaining a freight cost spike.
        Stick to ONLY the facts given below — do not add, infer, or invent anything.

        Route: {route}
        Week of: {week_of}
        Cost per tonne-km: {cost_per_tkm:.2f} INR
        vs own history: {own_pct}
        vs peer routes: {peer_pct}
        Matching note ID: {note_id}
        Note text: "{note_text}"

        Write one clear sentence that:
        - Names the note ID ({note_id}) and its date
        - Reflects the note's content (what happened, why costs rose)
        - Does NOT add any facts not in the note above
        Output only the sentence, no preamble.
    """).strip()


def _template_unexplained(
    route: str,
    week_of: pd.Timestamp,
    own_pct: str,
    peer_pct: str,
    closest_note_id: str,
    closest_note_text: str,
) -> str:
    """
    Deterministic template for unexplained (Yes) rows.
    If a closest note exists, mention why it does not qualify.
    """
    if closest_note_id:
        return (
            f"The closest note ({closest_note_id}) does not justify this cost rise "
            f"(it either does not apply to {route}, does not cover week {week_of.date()}, "
            f"or does not describe a cost increase). No validated explanation found; "
            f"flagged for human review."
        )
    return (
        f"No matching note found for {route} in week {week_of.date()}. "
        f"Cost rise looks unexplained and worth a human review."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reasons(
    validated: pd.DataFrame,
    structured_notes: pd.DataFrame,
    raw_notes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'reason' column to the validated DataFrame.

    For 'No (justified)' rows: call LLM (with cache) to write the reason.
    For 'Yes' rows: use deterministic template.

    Parameters
    ----------
    validated       : DataFrame with verdict, matched_note_id, closest_note_id
    structured_notes: structured notes DataFrame
    raw_notes       : raw notes with full 'note' text
    """
    provider = os.environ.get("LLM_PROVIDER", "none").lower()
    model    = os.environ.get("OLLAMA_MODEL", "llama3.2")
    if provider == "groq":
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif provider == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    # Build lookup for raw note text
    note_text_map = raw_notes.set_index("note_id")["note"].to_dict()

    reasons = []
    for _, row in validated.iterrows():
        if row["verdict"] == "No (justified)":
            nid = row["matched_note_id"]
            note_text = note_text_map.get(nid, "")
            own_val  = row.get("own_pct_diff")
            peer_val = row.get("peer_pct_diff")
            own_str  = row.get("vs_own_history", "N/A")
            peer_str = row.get("vs_similar_routes", "N/A")

            if provider == "none":
                # Deterministic fallback
                reason = (
                    f"Matches note {nid}: {note_text[:180].rstrip('.')}. "
                    f"The cost rise has a clear explanation."
                )
            else:
                prompt = _prompt_justified(
                    route=row["route"],
                    week_of=str(row["week_of"].date()),
                    cost_per_tkm=row["cost_per_tonne_km"],
                    own_pct=own_str,
                    peer_pct=peer_str,
                    note_id=nid,
                    note_text=note_text,
                )
                params = {"temperature": 0, "seed": 42}
                ck = _cache_key(model, prompt, params)
                cached = _cache_get(ck)

                if cached is not None:
                    reason = cached
                    _token_log.append({"source": "cache", "note_id": nid,
                                       "route": row["route"]})
                else:
                    try:
                        reason = _call_llm(prompt, provider, model)
                        _cache_put(ck, reason)
                        _token_log.append({"source": "live", "note_id": nid,
                                           "route": row["route"],
                                           "prompt_chars": len(prompt)})
                    except Exception as e:
                        # Fallback to template on any LLM error
                        reason = (
                            f"Matches note {nid}: {note_text[:180].rstrip('.')}. "
                            f"[LLM error: {e}]"
                        )
        else:
            # "Yes" — deterministic template, no LLM
            closest_id   = row.get("closest_note_id", "")
            closest_text = note_text_map.get(closest_id, "") if closest_id else ""
            reason = _template_unexplained(
                route=row["route"],
                week_of=row["week_of"],
                own_pct=row.get("vs_own_history", "N/A"),
                peer_pct=row.get("vs_similar_routes", "N/A"),
                closest_note_id=closest_id,
                closest_note_text=closest_text,
            )

        reasons.append(reason)

    result = validated.copy()
    result["reason"] = reasons
    return result
