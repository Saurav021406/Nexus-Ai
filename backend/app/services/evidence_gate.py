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

import re

DEFAULT_MIN_SIMILARITY = 0.35  # cosine similarity, 0-1 range (all-MiniLM-L6-v2)
DEFAULT_MIN_CHUNKS = 1

# A generic/summarization-style question ("what does this document say",
# "give me an overview", "summarize this") has no single sharp semantic
# target to embed against - it legitimately scores low similarity even
# against genuinely the best chunks available, since "summarize
# everything" isn't semantically close to any one specific passage the
# way a factual question is. Applying the similarity threshold to these
# would wrongly reject a question the document CAN answer, just not via a
# precise semantic match. If retrieval found any chunks at all for a
# query like this, that's enough evidence to let the LLM attempt an
# answer - the similarity-threshold check below is skipped, not the
# has-any-chunks check.
_GENERIC_SUMMARY_WORDS = {
    "summarize", "summary", "summarise", "overview", "explain", "describe",
    "gist", "content", "contents", "about", "topic", "topics", "cover",
    "covers", "say", "says", "said", "talk", "talks", "discuss",
    "discusses", "main", "point", "points", "highlights",
}


def _is_generic_summary_query(query: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", (query or "").lower()))
    return bool(words & _GENERIC_SUMMARY_WORDS)


def check_evidence(
    chunks: list[dict],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_chunks: int = DEFAULT_MIN_CHUNKS,
    query: str = "",
) -> dict:
    """Returns {"has_evidence": bool, "reason": str, "best_similarity": float | None}.

    `reason` is written to be shown directly to the user when has_evidence
    is False - Section 4's exact expected message shape ("mujhe iska jawab
    document mein nahi mila").

    `query` is optional and only used to detect generic/summarization-style
    questions (see _is_generic_summary_query) - passing "" preserves the
    original score-only behavior exactly."""
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
    if (
        best_similarity is not None
        and best_similarity < min_similarity
        and not _is_generic_summary_query(query)
    ):
        return {
            "has_evidence": False,
            "reason": (
                "I found some content in the document, but it doesn't look closely related "
                "enough to your question to answer confidently."
            ),
            "best_similarity": best_similarity,
        }

    return {"has_evidence": True, "reason": "", "best_similarity": best_similarity}
