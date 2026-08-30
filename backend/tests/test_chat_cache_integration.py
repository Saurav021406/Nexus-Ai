import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.routers.chat as chat_router
import app.services.cache as cache_module
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _reset_cache_and_auth():
    cache_module.clear_cache()
    app.dependency_overrides[chat_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()
    cache_module.clear_cache()


class FakeConsensusResult:
    def __init__(self, answer: str):
        self.answer = answer

    def to_meta_dict(self) -> dict:
        return {"models_used": ["fake"]}


def test_second_identical_tabular_question_hits_cache(monkeypatch):
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    monkeypatch.setattr(chat_router, "is_document_dataset", lambda dataset_id, user_id: False)
    monkeypatch.setattr(chat_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(chat_router, "build_data_summary", lambda dataframe: "revenue mean is 2")

    call_count = {"n": 0}

    def fake_consensus(prompt, **kwargs):
        call_count["n"] += 1
        return FakeConsensusResult("The average revenue is 2.")

    monkeypatch.setattr(chat_router, "get_consensus", fake_consensus)

    client = TestClient(app)
    payload = {"dataset_id": "d1", "question": "what is the average revenue", "history": []}

    first = client.post("/chat", json=payload)
    second = client.post("/chat", json=payload)

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]
    assert call_count["n"] == 1  # consensus only actually ran once


def test_out_of_domain_answer_is_never_cached(monkeypatch):
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    monkeypatch.setattr(chat_router, "is_document_dataset", lambda dataset_id, user_id: False)
    monkeypatch.setattr(chat_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(chat_router, "build_data_summary", lambda dataframe: "revenue mean is 2")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_consensus should never be called for an out-of-domain question")

    monkeypatch.setattr(chat_router, "get_consensus", fail_if_called)

    client = TestClient(app)
    payload = {"dataset_id": "d1", "question": "what is the capital of France", "history": []}

    first = client.post("/chat", json=payload)
    second = client.post("/chat", json=payload)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is False  # never cached, so still a miss both times


def test_question_with_history_bypasses_cache(monkeypatch):
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    monkeypatch.setattr(chat_router, "is_document_dataset", lambda dataset_id, user_id: False)
    monkeypatch.setattr(chat_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(chat_router, "build_data_summary", lambda dataframe: "revenue mean is 2")

    call_count = {"n": 0}

    def fake_consensus(prompt, **kwargs):
        call_count["n"] += 1
        return FakeConsensusResult("The average revenue is 2.")

    monkeypatch.setattr(chat_router, "get_consensus", fake_consensus)

    client = TestClient(app)
    payload = {
        "dataset_id": "d1",
        "question": "what about that",
        "history": [{"role": "user", "content": "what is the average revenue"}],
    }

    client.post("/chat", json=payload)
    client.post("/chat", json=payload)

    assert call_count["n"] == 2  # never cached because history was present
