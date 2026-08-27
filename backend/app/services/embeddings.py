"""Local embeddings (Step 3 of the RAG design, part 2 - chunking lives in
services/chunking.py).

    har chunk ka embedding banao (local model se, free)

Uses sentence-transformers with all-MiniLM-L6-v2 (384 dimensions) - a small,
fast, well-established model that runs entirely on the server with no
per-call API cost and no API key. This is a deliberate choice over an
API-based embedding provider (OpenAI, Cohere, etc.): embeddings get
generated for every chunk of every uploaded document, which would add up
in API cost and an extra network dependency for something that doesn't
need LLM-level reasoning.

The model is lazy-loaded (only downloaded/loaded into memory on first real
use, not at server startup) so the app doesn't pay that cost - a few
hundred MB of weights, downloaded once from Hugging Face and cached locally
- for requests that never touch document embeddings at all.
"""

from __future__ import annotations

from typing import Any

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

_model: Any = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Returns one 384-dim embedding vector per input text, same order.
    Batches internally (sentence-transformers handles this efficiently) -
    callers should pass all chunks for a document in one call rather than
    looping and calling this once per chunk."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """Convenience wrapper for embedding a single search query (Step 4:
    hybrid retrieval will call this against the same model/dimensions used
    at ingestion time - vectors from different models aren't comparable,
    so this MUST stay in sync with embed_texts())."""
    return embed_texts([query])[0]
