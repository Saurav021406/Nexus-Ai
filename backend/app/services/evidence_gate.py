"""Evidence Gate (Step 5 of the RAG design).

    retrieved chunks (Step 4)
        |
    Evidence Gate    <- this file. NO LLM call. Pure score checking.
        |
   +----+----+
   |         |
  NO        YES
   |         |
 "mujhe iska    continue to answer
  jawab document  generation (a later
  mein nahi        step - not built
  mila"            here)
 (no LLM call)

The whole point: if the retrieved chunks aren't actually relevant, don't
let an LLM "answer anyway" and hallucinate something plausible-sounding
from a document that doesn't actually contain the answer. Reject before
spending a token, same philosophy as Domain Gate (Step 1) - the cheapest,
most honest response to "I don't know" is saying so, not guessing.

Chunks coming out of hybrid_search() (services/retrieval.py) may or may
not carry a "similarity" field, depending on which search path(s) found
them - Reciprocal Rank Fusion keeps whichever list an item was first seen
in, so a chunk found only by keyword search won't have a similarity score
at all (it'll have "rank" instead). This gate accounts for that: a missing
similarity score isn't treated as automatic failure - an exact keyword
match is itself real evidence, even without a comparable similarity number.
"""

from __future__ import annotations

DEFAULT_MIN_SIMILARITY = 0.35  # cosine similarity, 0-1 range (all-MiniLM-L6-v2)
DEFAULT_MIN_CHUNKS = 1


def check_evidence(
    chunks: list[dict],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
) -> dict:
    """Returns {"has_evidence": bool, "reason": str, "best_similarity": float | None}.

    `reason` is written to be shown directly to the user when has_evidence
    is False - Section 4's exact expected message shape ("mujhe iska jawab
    document mein nahi mila")."""
    if not chunks:
        return {
            "has_evidence": False,
            "reason": "I couldn't find anything relevant to this question in the document.",
            "best_similarity": None,
        }

    if len(chunks) < min_chunks:
        return {
            "has_evidence": False,
            "reason": "Not enough matching content was found in the document to answer confidently.",
            "best_similarity": None,
        }

    similarities = [c["similarity"] for c in chunks if c.get("similarity") is not None]
    best_similarity = max(similarities) if similarities else None

    # A missing similarity score (chunk found only via keyword search) is
    # not itself a failure - an exact word match is real evidence on its
    # own. Only an ACTUAL low similarity score is grounds for rejection.
    if best_similarity is not None and best_similarity < min_similarity:
        return {
            "has_evidence": False,
            "reason": (
                "I found some content in the document, but it doesn't look closely related "
                "enough to your question to answer confidently."
            ),
            "best_similarity": best_similarity,
        }

    return {"has_evidence": True, "reason": "", "best_similarity": best_similarity}
