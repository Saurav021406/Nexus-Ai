import types

import app.services.document_chunks as document_chunks_module
from app.services.chunking import chunk_text


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name):
        self.name = name
        self._mode = None
        self._delete_filters: dict = {}

    def insert(self, rows):
        self._inserted = rows
        self._mode = "insert"
        return self

    def delete(self):
        self._mode = "delete"
        return self

    def eq(self, col, val):
        self._delete_filters[col] = val
        return self

    def execute(self):
        if self._mode == "insert":
            return _FakeResult(self._inserted)
        return _FakeResult([])


class _FakeSupabase:
    def __init__(self):
        self.last_table = None

    def table(self, name):
        self.last_table = _FakeTable(name)
        return self.last_table


def _fake_embed(texts):
    return [[0.1, 0.2, 0.3] for _ in texts]


def test_ingest_stores_one_row_per_chunk_with_correct_shape(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(document_chunks_module, "supabase_admin", fake_supabase)
    monkeypatch.setattr(document_chunks_module, "embed_texts", _fake_embed)

    count = document_chunks_module.ingest_document_chunks("dataset-1", "user-1", "short document text here")

    assert count == 1
    row = fake_supabase.last_table._inserted[0]
    assert row["dataset_id"] == "dataset-1"
    assert row["user_id"] == "user-1"
    assert row["chunk_index"] == 0
    assert row["chunk_text"] == "short document text here"
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["word_count"] == 4


def test_ingest_chunk_count_matches_chunking_module_output(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(document_chunks_module, "supabase_admin", fake_supabase)
    monkeypatch.setattr(document_chunks_module, "embed_texts", _fake_embed)

    long_text = " ".join(f"sentence{i} about important details" for i in range(600))
    count = document_chunks_module.ingest_document_chunks("dataset-2", "user-1", long_text)

    expected = len(chunk_text(long_text))
    assert count == expected
    assert count > 1  # sanity check that this text actually needed multiple chunks


def test_ingest_empty_text_returns_zero_without_calling_embed(monkeypatch):
    calls = {"n": 0}

    def trap_embed(texts):
        calls["n"] += 1
        return _fake_embed(texts)

    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(document_chunks_module, "supabase_admin", fake_supabase)
    monkeypatch.setattr(document_chunks_module, "embed_texts", trap_embed)

    count = document_chunks_module.ingest_document_chunks("dataset-3", "user-1", "")

    assert count == 0
    assert calls["n"] == 0, "embedding should never be called for empty text"


def test_delete_document_chunks_filters_by_dataset_and_user(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(document_chunks_module, "supabase_admin", fake_supabase)

    document_chunks_module.delete_document_chunks("dataset-1", "user-1")

    assert fake_supabase.last_table._delete_filters == {"dataset_id": "dataset-1", "user_id": "user-1"}
