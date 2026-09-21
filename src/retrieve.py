"""
retrieve.py — Retrieval of candidate notes for a candidate row.

Strategy:
  1. Try sentence-transformers (all-MiniLM-L6-v2) for semantic search.
  2. Fall back to TF-IDF cosine similarity if sentence-transformers is unavailable.
  3. Always include ALL-routes notes as candidates regardless of score.

The retriever returns the top-k note IDs (strings) for downstream validation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

# Try to import sentence-transformers; fall back to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Module-level cache for the model (loaded once)
_st_model: Optional[object] = None
_tfidf_vectorizer: Optional[object] = None
_tfidf_matrix: Optional[object] = None
_notes_for_retrieval: Optional[pd.DataFrame] = None


def _get_st_model():
    global _st_model
    if _st_model is None and _ST_AVAILABLE:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _build_tfidf(notes: pd.DataFrame):
    """Build TF-IDF matrix over note texts."""
    global _tfidf_vectorizer, _tfidf_matrix, _notes_for_retrieval
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_matrix     = _tfidf_vectorizer.fit_transform(notes["note_text"])
    _notes_for_retrieval = notes.reset_index(drop=True)


def build_retriever(notes_df: pd.DataFrame, raw_notes: pd.DataFrame) -> None:
    """
    Initialise retrieval index.

    notes_df    : structured notes (from notes.py load_structured_notes)
    raw_notes   : raw notes with full 'note' text column
    """
    global _notes_for_retrieval

    # Join full note text onto structured notes
    merged = notes_df.merge(
        raw_notes[["note_id", "note"]].rename(columns={"note": "note_text"}),
        on="note_id",
        how="left",
    )
    _notes_for_retrieval = merged.reset_index(drop=True)

    if not _ST_AVAILABLE:
        # Fall back to TF-IDF
        try:
            _build_tfidf(_notes_for_retrieval)
        except ImportError:
            pass  # No retriever available; we'll use brute-force text search


def retrieve_candidates(
    route: str,
    week_of: pd.Timestamp,
    own_pct: float | None,
    peer_pct: float | None,
    top_k: int = 5,
) -> list[str]:
    """
    Return a list of candidate note_ids for the given route/week.

    Always includes ALL-routes notes.  Then scores the rest by similarity
    to a query string combining route, week and cost movement.
    """
    if _notes_for_retrieval is None:
        return []

    notes = _notes_for_retrieval.copy()

    # Build query
    own_str  = f"{own_pct*100:+.1f}%" if own_pct is not None and not np.isnan(own_pct) else "N/A"
    peer_str = f"{peer_pct*100:+.1f}%" if peer_pct is not None and not np.isnan(peer_pct) else "N/A"
    query = (
        f"Route {route} week {week_of.date()} "
        f"cost increased {own_str} vs own history {peer_str} vs peers"
    )

    # Always include ALL-routes notes
    all_notes = notes[notes["routes"] == "ALL"]["note_id"].tolist()

    # Also always include route-specific notes for this route
    route_specific = notes[notes["routes"] == route]["note_id"].tolist()
    always_include = sorted(set(all_notes + route_specific))

    # Semantic/TF-IDF ranking of remaining notes
    remaining = notes[~notes["note_id"].isin(always_include)]

    ranked: list[str] = []
    if len(remaining) > 0:
        if _ST_AVAILABLE and _get_st_model() is not None:
            model = _get_st_model()
            query_vec = model.encode([query])
            note_vecs = model.encode(remaining["note_text"].tolist())
            sims = np.dot(note_vecs, query_vec.T).flatten()
            top_idx = np.argsort(sims)[::-1][: max(0, top_k - len(always_include))]
            ranked = remaining.iloc[top_idx]["note_id"].tolist()
        elif _tfidf_vectorizer is not None and _tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = _tfidf_vectorizer.transform([query])
            # Only score remaining notes
            remaining_idx = remaining.index.tolist()
            remaining_matrix = _tfidf_matrix[remaining_idx]
            sims = cosine_similarity(q_vec, remaining_matrix).flatten()
            top_idx = np.argsort(sims)[::-1][: max(0, top_k - len(always_include))]
            ranked = remaining.iloc[top_idx]["note_id"].tolist()

    return always_include + ranked
