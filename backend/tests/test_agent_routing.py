import pandas as pd
import pytest

import app.routers.agent as agent_router


class FakePayload:
    def __init__(self, dataset_id="d1", query="what is the average revenue"):
        self.dataset_id = dataset_id
        self.query = query


def test_out_of_domain_tabular_query_is_rejected_before_manager_runs(monkeypatch):
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    monkeypatch.setattr(agent_router, "is_document_dataset", lambda dataset_id, user_id: False)
    monkeypatch.setattr(agent_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(agent_router, "build_data_summary", lambda dataframe: "revenue stats")

    with pytest.raises(agent_router.RoutingRejected):
        agent_router._build_initial_state(FakePayload(query="what is the capital of France"), "u1")


def test_in_domain_tabular_query_builds_state_normally(monkeypatch):
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    monkeypatch.setattr(agent_router, "is_document_dataset", lambda dataset_id, user_id: False)
    monkeypatch.setattr(agent_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(agent_router, "build_data_summary", lambda dataframe: "revenue mean is 2")
    monkeypatch.setattr(agent_router, "get_dataset_record", lambda dataset_id, user_id: {"filename": "sales.csv"})

    state = agent_router._build_initial_state(FakePayload(query="what is the average revenue"), "u1")

    assert state.data_summary == "revenue mean is 2"
    assert state.dataset_columns == ["revenue"]


def test_document_dataset_with_no_evidence_is_rejected_before_manager_runs(monkeypatch):
    monkeypatch.setattr(agent_router, "is_document_dataset", lambda dataset_id, user_id: True)
    monkeypatch.setattr(agent_router, "get_document_text", lambda dataset_id, user_id: "refund policy details")
    monkeypatch.setattr(
        agent_router,
        "hybrid_search",
        lambda dataset_id, user_id, query, top_k: [{"id": "c1", "chunk_text": "unrelated", "similarity": 0.1}],
    )
    monkeypatch.setattr(agent_router, "rerank_chunks", lambda query, candidates, top_k: candidates[:top_k])

    with pytest.raises(agent_router.RoutingRejected):
        agent_router._build_initial_state(FakePayload(query="what is the refund policy"), "u1")


def test_document_dataset_with_evidence_builds_state_with_excerpt_context(monkeypatch):
    monkeypatch.setattr(agent_router, "is_document_dataset", lambda dataset_id, user_id: True)
    monkeypatch.setattr(agent_router, "get_document_text", lambda dataset_id, user_id: "refund policy details")
    monkeypatch.setattr(
        agent_router,
        "hybrid_search",
        lambda dataset_id, user_id, query, top_k: [
            {"id": "c1", "chunk_index": 0, "chunk_text": "refunds within 30 days", "similarity": 0.8}
        ],
    )
    monkeypatch.setattr(agent_router, "rerank_chunks", lambda query, candidates, top_k: candidates[:top_k])

    state = agent_router._build_initial_state(FakePayload(query="what is the refund policy"), "u1")

    assert "refunds within 30 days" in state.data_summary
    assert state.classification["primary_domain"] == "Document"
    assert state.dataset_columns == []


def test_document_out_of_domain_query_never_calls_hybrid_search(monkeypatch):
    monkeypatch.setattr(agent_router, "is_document_dataset", lambda dataset_id, user_id: True)
    monkeypatch.setattr(agent_router, "get_document_text", lambda dataset_id, user_id: "refund policy details")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("hybrid_search should never run for an out-of-domain query")

    monkeypatch.setattr(agent_router, "hybrid_search", fail_if_called)

    with pytest.raises(agent_router.RoutingRejected):
        agent_router._build_initial_state(FakePayload(query="what is the capital of France"), "u1")


def test_rejected_result_shape_matches_manager_v2_blocked_shape():
    result = agent_router._rejected_result("out of domain")
    assert result["status"] == "rejected"
    assert result["result"]["error"] == "out of domain"
    assert result["approval"] is None
