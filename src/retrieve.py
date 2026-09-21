"""
retrieve.py — Retrieval of candidate notes for a candidate row.

Strategy (priority order):
  1. ChromaDB (vector database) with all-MiniLM-L6-v2 embeddings — persistent collection
  2. Sentence-transformers in-memory semantic search (fallback if ChromaDB unavailable)
  3. TF-IDF cosine similarity (final fallback)

The retriever returns the top-k note IDs (strings) for downstream validation.
Similarity scores are logged to allow visibility into the RAG pipeline.

Usage:
  build_retriever(notes_df, raw_notes)
  candidates = retrieve_candidates(route, week_of, own_pct, peer_pct, verbose=True)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# ── Sentence-Transformers ────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ── Module-level state ───────────────────────────────────────────────────────
_chroma_collection: Optional[object] = None       # ChromaDB collection
_chroma_note_ids:   Optional[list[str]] = None    # all IDs in the collection
_st_model:          Optional[object] = None
_st_notes:          Optional[pd.DataFrame] = None
_tfidf_vectorizer:  Optional[object] = None
_tfidf_matrix:      Optional[object] = None
_notes_for_retrieval: Optional[pd.DataFrame] = None

CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma_store"


def _get_st_model() -> object:
    global _st_model
    if _st_model is None and _ST_AVAILABLE:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _build_tfidf(notes: pd.DataFrame) -> None:
    """Build TF-IDF matrix over note texts."""
    global _tfidf_vectorizer, _tfidf_matrix, _notes_for_retrieval
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_matrix     = _tfidf_vectorizer.fit_transform(notes["note_text"])
    _notes_for_retrieval = notes.reset_index(drop=True)


def _build_chroma(notes: pd.DataFrame) -> bool:
    """
    Build a ChromaDB collection with sentence-transformer embeddings.
    Returns True if successful.
    """
    global _chroma_collection, _chroma_note_ids

    if not _CHROMA_AVAILABLE:
        return False

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # Use sentence-transformers embedding function if available
        if _ST_AVAILABLE:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            ef = embedding_functions.DefaultEmbeddingFunction()

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Delete existing collection to rebuild fresh from current notes
        try:
            client.delete_collection("freight_notes")
        except Exception:
            pass

        collection = client.create_collection(
            name="freight_notes",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        ids       = notes["note_id"].tolist()
        documents = notes["note_text"].tolist()
        metadatas = [
            {
                "routes":    str(r),
                "direction": str(d),
                "start_date": str(s),
            }
            for r, d, s in zip(
                notes["routes"], notes["direction"], notes["start_date"]
            )
        ]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        _chroma_collection = collection
        _chroma_note_ids   = ids
        return True

    except Exception as e:
        print(f"[retrieve] ChromaDB build failed ({e}); falling back to TF-IDF.")
        return False


def build_retriever(notes_df: pd.DataFrame, raw_notes: pd.DataFrame) -> None:
    """
    Initialise retrieval index.

    notes_df  : structured notes (from notes.py load_structured_notes)
    raw_notes : raw notes with full 'note' text column
    """
    global _notes_for_retrieval

    # Join full note text onto structured notes
    merged = notes_df.merge(
        raw_notes[["note_id", "note"]].rename(columns={"note": "note_text"}),
        on="note_id",
        how="left",
    )
    _notes_for_retrieval = merged.reset_index(drop=True)

    # Priority 1: ChromaDB
    if _build_chroma(_notes_for_retrieval):
        print("[retrieve] Using ChromaDB (vector database) for semantic retrieval.")
        return

    # Priority 2: sentence-transformers in-memory
    if _ST_AVAILABLE:
        global _st_notes
        _st_notes = _notes_for_retrieval.copy()
        print("[retrieve] Using sentence-transformers for in-memory semantic search.")
        return

    # Priority 3: TF-IDF
    try:
        _build_tfidf(_notes_for_retrieval)
        print("[retrieve] Using TF-IDF cosine similarity (install chromadb for vector DB).")
    except ImportError:
        print("[retrieve] WARNING: No retriever available. Note ranking disabled.")


def retrieve_candidates(
    route: str,
    week_of: pd.Timestamp,
    own_pct: float | None,
    peer_pct: float | None,
    top_k: int = 5,
    verbose: bool = False,
) -> list[str]:
    """
    Return a list of candidate note_ids for the given route/week.

    Always includes ALL-routes notes and route-specific notes first.
    Then ranks remaining notes by semantic similarity to a query string.

    Parameters
    ----------
    route    : e.g. "Chennai-Bangalore"
    week_of  : Monday timestamp
    own_pct  : pct deviation from own history (float or NaN)
    peer_pct : pct deviation from peer routes (float or NaN)
    top_k    : total candidates to return
    verbose  : if True, print similarity scores (useful for demo)
    """
    if _notes_for_retrieval is None:
        return []

    notes = _notes_for_retrieval.copy()

    # Build query string
    own_str  = f"{own_pct*100:+.1f}%" if own_pct is not None and not np.isnan(own_pct) else "N/A"
    peer_str = f"{peer_pct*100:+.1f}%" if peer_pct is not None and not np.isnan(peer_pct) else "N/A"
    query = (
        f"Route {route} week {week_of.date()} "
        f"freight cost increased {own_str} vs own history {peer_str} vs peer routes"
    )

    # Always include ALL-routes notes + route-specific notes
    all_notes      = notes[notes["routes"] == "ALL"]["note_id"].tolist()
    route_specific = notes[notes["routes"] == route]["note_id"].tolist()
    always_include = sorted(set(all_notes + route_specific))
    remaining      = notes[~notes["note_id"].isin(always_include)]

    ranked_with_scores: list[tuple[str, float]] = []

    if len(remaining) > 0:
        # ── ChromaDB retrieval ──────────────────────────────────────────────
        if _chroma_collection is not None:
            try:
                exclude_ids = set(always_include)
                n_results   = min(top_k, max(1, len(remaining)))
                results     = _chroma_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"routes": {"$nin": list(exclude_ids)}} if False else None,
                )
                chroma_ids    = results["ids"][0]
                chroma_dists  = results["distances"][0]  # cosine distance

                for nid, dist in zip(chroma_ids, chroma_dists):
                    if nid not in always_include:
                        sim = 1.0 - dist  # convert distance to similarity
                        ranked_with_scores.append((nid, round(sim, 4)))

            except Exception:
                pass  # fall through to next method

        # ── Sentence-transformers in-memory ─────────────────────────────────
        if not ranked_with_scores and _ST_AVAILABLE and _get_st_model() is not None:
            model      = _get_st_model()
            query_vec  = model.encode([query])
            note_vecs  = model.encode(remaining["note_text"].tolist())
            sims       = np.dot(note_vecs, query_vec.T).flatten()
            top_n      = min(top_k - len(always_include), len(remaining))
            top_idx    = np.argsort(sims)[::-1][:max(0, top_n)]
            for i in top_idx:
                ranked_with_scores.append((remaining.iloc[i]["note_id"], round(float(sims[i]), 4)))

        # ── TF-IDF fallback ─────────────────────────────────────────────────
        if not ranked_with_scores and _tfidf_vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            q_vec          = _tfidf_vectorizer.transform([query])
            remaining_idx  = remaining.index.tolist()
            remaining_mat  = _tfidf_matrix[remaining_idx]
            sims           = cos_sim(q_vec, remaining_mat).flatten()
            top_n          = max(0, top_k - len(always_include))
            top_idx        = np.argsort(sims)[::-1][:top_n]
            for i in top_idx:
                ranked_with_scores.append((remaining.iloc[i]["note_id"], round(float(sims[i]), 4)))

    ranked = [nid for nid, _ in ranked_with_scores]

    if verbose and ranked_with_scores:
        print(f"\n[RAG] Query: {query[:80]}...")
        print(f"[RAG] Always-include: {always_include}")
        print(f"[RAG] Ranked candidates (with similarity scores):")
        for nid, score in ranked_with_scores:
            print(f"      {nid}: {score:.4f}")

    return always_include + ranked
