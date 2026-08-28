import app.services.retrieval as retrieval_module
from app.services.evidence_gate import check_evidence


def test_no_chunks_means_no_evidence():
    result = check_evidence([])
    assert result["has_evidence"] is False
    assert result["best_similarity"] is None


def test_high_similarity_chunk_passes():
    result = check_evidence([{"id": "c1", "similarity": 0.8, "chunk_text": "relevant"}])
    assert result["has_evidence"] is True
    assert result["best_similarity"] == 0.8


def test_low_similarity_chunk_fails():
    result = check_evidence([{"id": "c1", "similarity": 0.1, "chunk_text": "barely related"}])
    assert result["has_evidence"] is False
    assert result["best_similarity"] == 0.1


def test_keyword_only_chunk_with_no_similarity_field_still_passes():
    """A chunk found only via keyword search carries 'rank', not
    'similarity' - a missing similarity score must not be treated as
    automatic rejection, since an exact word match is real evidence too."""
    result = check_evidence([{"id": "c1", "rank": 0.5, "chunk_text": "exact keyword match"}])
    assert result["has_evidence"] is True
    assert result["best_similarity"] is None


def test_mixed_chunks_best_similarity_is_the_max_of_those_present():
    result = check_evidence([
        {"id": "c1", "chunk_text": "keyword only"},
        {"id": "c2", "similarity": 0.6, "chunk_text": "vector match"},
        {"id": "c3", "similarity": 0.9, "chunk_text": "best vector match"},
    ])
    assert result["has_evidence"] is True
    assert result["best_similarity"] == 0.9


def test_min_chunks_threshold_is_respected():
    result = check_evidence([{"id": "c1", "similarity": 0.8}], min_chunks=2)
    assert result["has_evidence"] is False


def test_custom_similarity_threshold_rejects_below_it():
    result = check_evidence([{"id": "c1", "similarity": 0.4}], min_similarity=0.5)
    assert result["has_evidence"] is False


def test_custom_similarity_threshold_accepts_at_or_above_it():
    result = check_evidence([{"id": "c1", "similarity": 0.4}], min_similarity=0.3)
    assert result["has_evidence"] is True


def test_reason_message_is_present_and_user_facing_when_rejected():
    result = check_evidence([])
    assert result["reason"]
    assert "reason" not in ("", None)


def test_reason_is_empty_when_evidence_is_sufficient():
    result = check_evidence([{"id": "c1", "similarity": 0.8}])
    assert result["reason"] == ""


def test_real_hybrid_search_output_shape_is_handled_correctly(monkeypatch):
    """End-to-end shape check: hybrid_search's actual output (RRF-merged,
    with 'score' added, original 'similarity'/'rank' fields preserved from
    whichever list first found each chunk) must be directly consumable by
    check_evidence() without any reshaping in between."""
    def fake_vector(dataset_id, user_id, query, top_k):
        return [
            {"id": "chunk1", "chunk_text": "refund policy text", "similarity": 0.72},
            {"id": "chunk2", "chunk_text": "unrelated text", "similarity": 0.15},
        ]

    def fake_keyword(dataset_id, user_id, query, top_k):
        return [{"id": "chunk3", "chunk_text": "exact refund match", "rank": 0.8}]

    monkeypatch.setattr(retrieval_module, "vector_search", fake_vector)
    monkeypatch.setattr(retrieval_module, "keyword_search", fake_keyword)

    results = retrieval_module.hybrid_search("d1", "u1", "what is the refund policy", top_k=5)
    gate_result = check_evidence(results)

    assert gate_result["has_evidence"] is True
    assert gate_result["best_similarity"] == 0.72


def test_real_hybrid_search_rejection_case(monkeypatch):
    monkeypatch.setattr(
        retrieval_module, "vector_search",
        lambda dataset_id, user_id, query, top_k: [{"id": "c1", "similarity": 0.05, "chunk_text": "unrelated"}],
    )
    monkeypatch.setattr(retrieval_module, "keyword_search", lambda dataset_id, user_id, query, top_k: [])

    results = retrieval_module.hybrid_search("d1", "u1", "random unrelated question", top_k=5)
    gate_result = check_evidence(results)

    assert gate_result["has_evidence"] is False
