-- Document chunks + embeddings table (Step 3 of the RAG design: chunking +
-- embeddings + pgvector setup). Run this once in your Supabase SQL editor.
--
-- Requires the pgvector extension - Supabase projects have this available
-- but it needs enabling once per project (the first line does that).
--
-- Embedding dimension is 384, matching sentence-transformers'
-- all-MiniLM-L6-v2 (see backend/app/services/embeddings.py). If you ever
-- change the embedding model to one with a different output size, this
-- column's dimension must change to match - vectors from a different
-- model/dimension are never comparable to ones already stored here.

create extension if not exists vector;

create table if not exists document_chunks (
    id uuid primary key default gen_random_uuid(),
    dataset_id uuid not null,
    user_id uuid not null,
    chunk_index integer not null,
    chunk_text text not null,
    word_count integer not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_dataset_idx
    on document_chunks (dataset_id);

create index if not exists document_chunks_user_idx
    on document_chunks (user_id);

-- Approximate nearest-neighbor index for fast similarity search (Step 4:
-- hybrid retrieval will query against this). IVFFlat needs `lists` tuned
-- to roughly sqrt(row_count) once you have real data - 100 is a reasonable
-- starting point for a small/medium document corpus and can be rebuilt
-- later with `reindex index document_chunks_embedding_idx;` as it grows.
create index if not exists document_chunks_embedding_idx
    on document_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- If you use Supabase Row Level Security (same pattern as the other
-- migrations in this project):
--
-- alter table document_chunks enable row level security;
--
-- create policy "Users can manage their own document chunks"
--     on document_chunks
--     for all
--     using (auth.uid() = user_id)
--     with check (auth.uid() = user_id);
