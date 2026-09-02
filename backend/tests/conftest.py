"""Shared test setup.

Every test in this suite runs with fake credentials and a mocked Supabase
client - nothing here ever needs a real Supabase project, and nothing here
makes real calls to Groq/NVIDIA/OpenRouter (each test that needs an LLM
response monkeypatches the specific function it needs, the same pattern
used throughout Phase 4 development).

This file must set env vars and mock app.supabase_client BEFORE any test
module does `import app.something` - pytest loads conftest.py in a
directory before collecting sibling test files, which is what makes that
ordering guarantee hold.
"""

import os
import sys
import types

os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-key")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

_fake_supabase_module = types.ModuleType("app.supabase_client")
_fake_supabase_module.supabase_anon = object()
_fake_supabase_module.supabase_admin = object()
sys.modules["app.supabase_client"] = _fake_supabase_module

import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Global, autouse: services/rate_limiter.py keeps its request counts
    in a module-level dict that persists for the whole pytest session, not
    per-test. Without this reset, any test hitting a rate-limited endpoint
    (e.g. /chat, /agent/run, /automl/run) enough times across the FULL
    test suite - not just within its own test function - could start
    failing with 429s it has nothing to do with, purely because of
    execution order and how many other tests reused the same fake user id
    before it ran. Resetting before every test removes that cross-test
    coupling entirely."""
    from app.services import rate_limiter, usage_stats

    rate_limiter.reset_all()
    usage_stats.reset_all()
    yield
    rate_limiter.reset_all()
    usage_stats.reset_all()
