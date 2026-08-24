"""Domain Gate (Step 1 of the RAG/Domain Router design).

    User Query
        |
   Domain Gate    <- this file. NO LLM call. Free, instant.
        |
   +----+----+
   |         |
 OUT OF   TABULAR MATCH -> continue to Input Security -> Manager -> ...
 DOMAIN
   |
 reject immediately, zero token cost

Runs BEFORE Input Security and BEFORE the Manager - it's the cheapest,
fastest check in the whole pipeline (pure Python string matching, no
network call at all), so it goes first: rejecting an obviously off-topic
question here means Input Security's LLM call and the Manager's planning
LLM call never happen either.

Currently tabular-only, matching what this app actually ingests today
(CSV/Excel -> pandas). The document/RAG path (PDF/Word -> chunking ->
embeddings -> hybrid retrieval) doesn't exist yet - when it does, this
gate is exactly where the TABULAR MATCH / DOCUMENT MATCH branch belongs;
for now every dataset is tabular, so that branch is a no-op.

Deliberately permissive: a false negative (blocking a legitimate question)
is a worse user experience than a false positive (letting an odd-but-fine
question through to the Manager, which still has its own reasoning to
handle it sensibly). This is a coarse pre-filter, not a strict classifier.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "this", "that", "these", "those", "to", "of",
    "in", "on", "at", "for", "with", "about", "as", "by", "and", "or",
    "but", "if", "so", "than", "then", "there", "here", "what", "which",
    "who", "whom", "how", "me", "my", "can", "could", "would", "should",
    "will", "shall", "please", "give", "tell", "us", "our",
}

# Generic analysis vocabulary - words that make a query "about the dataset"
# no matter what the dataset's actual columns are. A query containing any
# of these is treated as in-domain automatically, since asking someone to
# use the literal column name just to get past a filter would be a bad
# experience (e.g. "summarize this" should always work).
_GENERIC_ANALYSIS_WORDS = {
    "analyze", "analysis", "analyse", "summary", "summarize", "summarise",
    "overview", "trend", "trends", "pattern", "patterns", "compare",
    "comparison", "correlation", "correlate", "distribution", "insight",
    "insights", "explain", "describe", "recommend", "recommendation",
    "recommendations", "report", "statistics", "stats", "data", "dataset",
    "column", "columns", "row", "rows", "average", "mean", "median",
    "total", "count", "chart", "graph", "visualize", "visualise",
    "forecast", "predict", "prediction", "anomaly", "outlier", "outliers",
    "sql", "query", "table",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _dataset_vocabulary(dataset_columns: list[str], data_summary: str) -> set[str]:
    vocab: set[str] = set()
    for col in dataset_columns:
        # split_words handles snake_case / camelCase / kebab-case column names
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", col).replace("_", " ").replace("-", " ")
        vocab |= _tokenize(spaced)
    vocab |= _tokenize(data_summary)
    return vocab


def check_domain_relevance(
    user_query: str, dataset_columns: list[str], data_summary: str
) -> dict:
    """Returns {"in_domain": bool, "reason": str}. No LLM call, no network
    call - pure string matching, safe to run on every request for free."""
    query_words = _tokenize(user_query)

    if not query_words:
        return {"in_domain": True, "reason": "Query too short to classify - defaulting to allow."}

    if query_words & _GENERIC_ANALYSIS_WORDS:
        return {"in_domain": True, "reason": "Query uses general analysis language."}

    vocabulary = _dataset_vocabulary(dataset_columns, data_summary)
    overlap = query_words & vocabulary
    if overlap:
        return {"in_domain": True, "reason": f"Query overlaps dataset vocabulary: {sorted(overlap)}"}

    return {
        "in_domain": False,
        "reason": (
            "This question doesn't appear related to this dataset's columns or content "
            f"({', '.join(dataset_columns[:8])}{'...' if len(dataset_columns) > 8 else ''})."
        ),
    }
