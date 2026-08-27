"""Document ingestion orchestration (Step 3 of the RAG design - ties
services/chunking.py and services/embeddings.py together and writes the
result to Supabase/pgvector).

    PDF/Word file -> text extract (Step 2, already done)
        -> chunk_text() -> embed_texts() -> store in document_chunks
        (all of this happens ONCE at upload time, not per query - see
        module docstring in chunking.py and the RAG design's own Step 1
        note: "INGESTION (upload ke time hi ek baar hota hai, query ke
        time nahi)")

Retrieval (searching these chunks at query time) is Step 4, a separate
piece of work - this module only writes, it never reads/searches.
"""

from __future__ import annotations

from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.supabase_client import supabase_admin

TABLE = "document_chunks"


def ingest_document_chunks(dataset_id: str, user_id: str, text: str) -> int:
    """Chunks `text`, embeds every chunk, and stores them all. Returns the
    number of chunks actually stored. Returns 0 (not an error) for empty
    text - an empty document has nothing to ingest, that's not a failure
    of this function."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts([c.text for c in chunks])

    rows = [
        {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "chunk_index": chunk.index,
            "chunk_text": chunk.text,
            "word_count": chunk.word_count,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    result = supabase_admin.table(TABLE).insert(rows).execute()
    return len(result.data) if result.data else 0


def delete_document_chunks(dataset_id: str, user_id: str) -> None:
    """Called when a document dataset is deleted - chunks shouldn't
    outlive the dataset they came from."""
    supabase_admin.table(TABLE).delete().eq("dataset_id", dataset_id).eq("user_id", user_id).execute()
