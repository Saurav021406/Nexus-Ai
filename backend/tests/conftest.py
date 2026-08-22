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
