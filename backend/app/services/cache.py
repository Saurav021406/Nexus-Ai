"""Query caching (Optimization Step 2).

In-memory TTL cache for chat answers, keyed on sha256(dataset_id +
normalized_question). Repeating the exact same question against the same
dataset - very common when a user re-reads an answer, refreshes, or two
people ask the same thing - returns instantly instead of re-running Domain
Gate, retrieval/reranking, and a consensus LLM call all over again.

In-memory rather than Redis/Supabase deliberately: this is a single-process
FastAPI deployment (see uvicorn in requirements.txt, no worker-pool /
external cache configured elsewhere in the codebase), so a plain dict with
a lock is simpler and has zero extra infra to run or fail. If/when this
runs behind multiple workers or processes, this should move to Supabase or
Redis so cache entries are shared - a plain dict is per-process only.

Only successful, in-domain, has-evidence answers are ever cached (callers'
responsibility - see routers/chat.py) - never a Domain Gate rejection or a
"not found in document" Evidence Gate answer. Those are cheap to reject
anyway (no LLM call), and caching them risks a fixed dataset returning a
stale "not found" after new content is uploaded, or a rejection reason
that no longer applies once column names change.
"""

from __future__ import annotations

import hashlib
import threading
import time

DEFAULT_TTL_SECONDS = 3600

_lock = threading.Lock()
_store: dict[str, tuple[float, dict]] = {}  # cache_key -> (expires_at_epoch, value)
_keys_by_dataset: dict[str, set[str]] = {}  # dataset_id -> set of cache_keys, for selective clear_cache()


def _normalize_question(question: str) -> str:
    """Collapse whitespace and case so trivially-different phrasings of the
    same question ("What's the total revenue?" vs "what's the total
    revenue ") still hit the same cache entry."""
    return " ".join(question.strip().lower().split())


def make_cache_key(dataset_id: str, question: str) -> str:
    normalized = _normalize_question(question)
    raw = f"{dataset_id}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(dataset_id: str, question: str) -> dict | None:
    """Returns the cached response dict if present and not expired, else
    None. Also lazily evicts the entry if it's found expired, so the store
    doesn't accumulate stale entries indefinitely between explicit
    clear_cache() calls."""
    key = make_cache_key(dataset_id, question)
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            _evict_locked(dataset_id, key)
            return None
        return value


def set_cached_response(dataset_id: str, question: str, response: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    key = make_cache_key(dataset_id, question)
    with _lock:
        _store[key] = (time.time() + ttl_seconds, response)
        _keys_by_dataset.setdefault(dataset_id, set()).add(key)


def _evict_locked(dataset_id: str, key: str) -> None:
    """Removes one entry from both _store and its dataset index. Caller
    must already hold _lock."""
    _store.pop(key, None)
    keys = _keys_by_dataset.get(dataset_id)
    if keys is not None:
        keys.discard(key)
        if not keys:
            del _keys_by_dataset[dataset_id]


def clear_cache(dataset_id: str | None = None) -> int:
    """Clears every cached entry for one dataset (if dataset_id is given)
    or the entire cache (if omitted) - call with a dataset_id after new
    content is uploaded to that dataset, since old cached answers may no
    longer reflect the current data. Returns the number of entries
    removed."""
    with _lock:
        if dataset_id is None:
            count = len(_store)
            _store.clear()
            _keys_by_dataset.clear()
            return count

        keys = _keys_by_dataset.pop(dataset_id, set())
        for key in keys:
            _store.pop(key, None)
        return len(keys)
