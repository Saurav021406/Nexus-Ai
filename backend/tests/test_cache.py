import time

import app.services.cache as cache_module


def setup_function():
    # Full reset between tests since the cache module holds module-level
    # state (a plain dict, by design - see cache.py docstring).
    cache_module.clear_cache()


def test_cache_miss_returns_none():
    assert cache_module.get_cached_response("d1", "what is the average revenue") is None


def test_cache_hit_returns_stored_response():
    response = {"answer": "The average revenue is 42.", "path": "tabular"}
    cache_module.set_cached_response("d1", "what is the average revenue", response)

    cached = cache_module.get_cached_response("d1", "what is the average revenue")

    assert cached == response


def test_cache_key_normalizes_whitespace_and_case():
    response = {"answer": "42"}
    cache_module.set_cached_response("d1", "  What Is The Average Revenue?  ", response)

    cached = cache_module.get_cached_response("d1", "what is the average revenue?")

    assert cached == response


def test_different_dataset_ids_do_not_share_cache_entries():
    cache_module.set_cached_response("d1", "average revenue", {"answer": "d1 answer"})
    cache_module.set_cached_response("d2", "average revenue", {"answer": "d2 answer"})

    assert cache_module.get_cached_response("d1", "average revenue") == {"answer": "d1 answer"}
    assert cache_module.get_cached_response("d2", "average revenue") == {"answer": "d2 answer"}


def test_expired_entry_is_not_returned():
    cache_module.set_cached_response("d1", "average revenue", {"answer": "42"}, ttl_seconds=0.01)
    time.sleep(0.05)

    assert cache_module.get_cached_response("d1", "average revenue") is None


def test_clear_cache_for_one_dataset_only_removes_that_dataset():
    cache_module.set_cached_response("d1", "q1", {"answer": "a1"})
    cache_module.set_cached_response("d2", "q1", {"answer": "a2"})

    removed = cache_module.clear_cache("d1")

    assert removed == 1
    assert cache_module.get_cached_response("d1", "q1") is None
    assert cache_module.get_cached_response("d2", "q1") == {"answer": "a2"}


def test_clear_cache_with_no_argument_clears_everything():
    cache_module.set_cached_response("d1", "q1", {"answer": "a1"})
    cache_module.set_cached_response("d2", "q1", {"answer": "a2"})

    removed = cache_module.clear_cache()

    assert removed == 2
    assert cache_module.get_cached_response("d1", "q1") is None
    assert cache_module.get_cached_response("d2", "q1") is None
