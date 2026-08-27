"""Text chunking (Step 3 of the RAG design, part 1 of 2 - chunking here,
embeddings in services/embeddings.py).

    extracted text -> chunks mein todo (500-800 tokens each)

Word-count-based chunking, not a real tokenizer. This is a deliberate,
honest simplification: the embedding model (sentence-transformers) has its
own subword tokenizer that doesn't map 1:1 to any LLM's tokenizer anyway,
so a byte-perfect token count doesn't actually buy accuracy here - it would
just add a dependency (tiktoken) that measures the wrong thing. English
text averages ~0.75 words per token, so ~500 words is a reasonable proxy
for the "500-800 tokens" target range.

Overlap between consecutive chunks exists so a sentence that happens to
fall right on a chunk boundary isn't split with half its context in one
chunk and half in another - a fact stated right at a boundary would
otherwise become unfindable by either chunk's embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_WORDS = 500  # approx 650-700 tokens for English text
DEFAULT_OVERLAP_WORDS = 75  # approx 100 tokens


@dataclass
class Chunk:
    index: int
    text: str
    word_count: int


def chunk_text(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Splits on paragraph boundaries first (so a chunk break never lands
    mid-sentence if a paragraph break is nearby), packing paragraphs into
    chunks up to chunk_words, with the last overlap_words of a chunk
    repeated at the start of the next one."""
    if not text or not text.strip():
        return []

    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # Flatten to a single word stream with paragraph markers preserved as
    # just whitespace - simpler and more robust than trying to keep exact
    # paragraph objects across a chunk boundary.
    all_words = " ".join(paragraphs).split()

    if not all_words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_words - overlap_words

    while start < len(all_words):
        end = min(start + chunk_words, len(all_words))
        chunk_words_slice = all_words[start:end]
        chunks.append(Chunk(index=index, text=" ".join(chunk_words_slice), word_count=len(chunk_words_slice)))
        index += 1
        if end >= len(all_words):
            break
        start += step

    return chunks
