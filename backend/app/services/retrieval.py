"""Hybrid Retrieval (Step 4 of the RAG design).

    Query aane par:
        -> Vector Search: match_document_chunks_vector() RPC (pgvector
           cosine similarity - semantic match)
        -> Keyword Search: match_document_chunks_keyword() RPC (Postgres
           full-text search via tsvector/GIN - exact term match)
        -> Dono ke results combine karo (hybrid, via Reciprocal Rank Fusion)

Two independent retrieval strategies exist because they fail in different,
complementary ways: vector search finds paraphrases and related concepts
but can miss an exact rare term (a product code, a proper noun); full-text
search finds exact terms but has no idea that "refund" and "money back"
mean the same thing. Combining both catches more real answers than either
alone, without needing a heavier cross-encoder reranker (Step 7, optional).

Both searches are plain Postgres functions (see
document_chunks_search_function.sql) called via .rpc() - neither pgvector's
`<=>` operator nor tsvector full-text matching is expressible through
Supabase's plain .select()/.eq() query builder, so the actual ranking work
happens in SQL, and this module just calls each function and merges the
two ranked lists it gets back.

Reciprocal Rank Fusion (RRF) is used to combine the two ranked lists rather
than trying to blend a cosine-similarity score with a full-text rank score
directly - those two numbers aren't on comparable scales, so averaging them
would be meaningless. RRF only looks at each item's RANK within its own
list, which sidesteps that problem entirely: score(item) = sum over the
lists it appears in of 1 / (k + rank), summed across lists. This is the
same technique Elasticsearch/OpenSearch use for hybrid search.

Callers should treat hybrid_search() as the only public entry point in
normal use - vector_search()/keyword_search() are exposed separately mainly
so tests (see test_evidence_gate.py) can monkeypatch either independently.
"""

from __future__ import annotations

from app.services.embeddings import embed_query
from app.supabase_client import supabase_admin

VECTOR_FUNCTION = "match_document_chunks_vector"
KEYWORD_FUNCTION = "match_document_chunks_keyword"
RRF_K = 60  # standard smoothing constant for Reciprocal Rank Fusion


def vector_search(dataset_id: str, user_id: str, query: str, top_k: int = 8) -> list[dict]:
    """Semantic search via the match_document_chunks_vector() Postgres
    function (pgvector cosine similarity)."""
    query_embedding = embed_query(query)

    result = supabase_admin.rpc(
        VECTOR_FUNCTION,
        {
            "query_embedding": query_embedding,
            "p_dataset_id": dataset_id,
            "p_user_id": user_id,
            "match_count": top_k,
        },
    ).execute()

    rows = result.data or []
    return [
        {
            "id": row["id"],
            "chunk_text": row["chunk_text"],
            "chunk_index": row.get("chunk_index"),
            "similarity": row["similarity"],
        }
        for row in rows
    ]


def keyword_search(dataset_id: str, user_id: str, query: str, top_k: int = 8) -> list[dict]:
    """Exact-term search via the match_document_chunks_keyword() Postgres
    function (native Postgres full-text search: to_tsvector/plainto_tsquery,
    backed by a GIN index) - real full-text ranking, not substring
    matching, so it scales to large documents without a table scan."""
    result = supabase_admin.rpc(
        KEYWORD_FUNCTION,
        {
            "query_text": query,
            "p_dataset_id": dataset_id,
            "p_user_id": user_id,
            "match_count": top_k,
        },
    ).execute()

    rows = result.data or []
    return [
        {
            "id": row["id"],
            "chunk_text": row["chunk_text"],
            "chunk_index": row.get("chunk_index"),
            "rank": row["rank"],
        }
        for row in rows
    ]


def hybrid_search(dataset_id: str, user_id: str, query: str, top_k: int = 8) -> list[dict]:
    """Runs vector_search() and keyword_search(), merges them with
    Reciprocal Rank Fusion, and returns the top_k chunks by combined score.

    A chunk found by only ONE of the two searches keeps whatever fields
    that search produced (similarity for vector-only, rank for
    keyword-only) - Evidence Gate (services/evidence_gate.py) is already
    written to treat a missing "similarity" as "not automatically a
    failure", exactly because of this. A chunk found by BOTH searches
    keeps the vector result's fields (checked first below), since a
    cosine similarity score is more informative than a keyword rank.
    """
    vector_results = vector_search(dataset_id, user_id, query, top_k=top_k * 2)
    keyword_results = keyword_search(dataset_id, user_id, query, top_k=top_k * 2)

    merged_chunks: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}

    for rank, chunk in enumerate(vector_results):
        chunk_id = chunk["id"]
        merged_chunks.setdefault(chunk_id, chunk)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, chunk in enumerate(keyword_results):
        chunk_id = chunk["id"]
        merged_chunks.setdefault(chunk_id, chunk)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    merged = [
        {**chunk, "score": rrf_scores[chunk_id]}
        for chunk_id, chunk in merged_chunks.items()
    ]
    merged.sort(key=lambda c: c["score"], reverse=True)
    return merged[:top_k]
