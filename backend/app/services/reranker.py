"""Reranking service for retrieved document chunks (Step 7).

Takes candidate chunks retrieved by Hybrid Search (vector + BM25/keyword)
and rescores them against the user query using a cross-encoder / reranker model.
"""

from __future__ import annotations

import re
from typing import Any

# Optional lightweight neural cross-encoder support
try:
    from flashrank import Ranker, RerankRequest

    _HAS_FLASHRANK = True
    _ranker: Ranker | None = None
except ImportError:
    _HAS_FLASHRANK = False
    _ranker = None


def _get_ranker() -> Any:
    global _ranker
    if _ranker is None and _HAS_FLASHRANK:
        # Nano cross-encoder (~4MB), runs in milliseconds on CPU
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    return _ranker


def _lexical_overlap_score(query: str, text: str) -> float:
    """Fallback lexical overlap scorer if flashrank is not installed."""
    q_words = set(re.findall(r"\w+", query.lower()))
    if not q_words:
        return 0.0
    t_words = set(re.findall(r"\w+", text.lower()))
    overlap = q_words.intersection(t_words)
    return len(overlap) / (len(q_words) + 1e-5)


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Reranks candidate chunks based on relevance to the query.

    Args:
        query: The user search query.
        chunks: List of candidate chunk dicts (each containing at least 'chunk_text').
        top_k: Number of highest-scoring chunks to keep.

    Returns:
        Sorted list of top_k chunk dicts with updated 'rerank_score'.
    """
    if not chunks or top_k <= 0:
        return []

    ranker = _get_ranker() if _HAS_FLASHRANK else None

    # Path A: Fast neural cross-encoder reranking
    if ranker:
        try:
            passages = [
                {"id": str(i), "text": c.get("chunk_text", "")}
                for i, c in enumerate(chunks)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            reranked = []
            for r in results[:top_k]:
                idx = int(r["id"])
                chunk_copy = dict(chunks[idx])
                chunk_copy["rerank_score"] = round(float(r.get("score", 0.0)), 4)
                reranked.append(chunk_copy)
            return reranked
        except Exception:
            # Gracefully fallback on any runtime error
            pass

    # Path B: Fallback (hybrid score + token overlap combination)
    scored = []
    for c in chunks:
        text = c.get("chunk_text", "")
        base_score = float(c.get("score", 0.5) or 0.5)
        overlap = _lexical_overlap_score(query, text)
        final_score = (base_score * 0.6) + (overlap * 0.4)

        chunk_copy = dict(c)
        chunk_copy["rerank_score"] = round(final_score, 4)
        scored.append(chunk_copy)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]