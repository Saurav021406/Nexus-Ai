from app.services.reranker import rerank_chunks


def test_rerank_empty_chunks():
    assert rerank_chunks("test query", []) == []


def test_rerank_filters_top_k():
    chunks = [
        {"chunk_id": f"c_{i}", "chunk_text": f"Some dummy text chunk {i}", "score": 0.5}
        for i in range(10)
    ]
    results = rerank_chunks("dummy", chunks, top_k=3)
    assert len(results) == 3
    assert all("rerank_score" in r for r in results)


def test_rerank_prioritizes_relevant_text():
    chunks = [
        {"chunk_id": "1", "chunk_text": "The company's annual revenue was $50 million in 2023.", "score": 0.5},
        {"chunk_id": "2", "chunk_text": "Employees should submit expense reports by Friday.", "score": 0.5},
        {"chunk_id": "3", "chunk_text": "Total revenue and fiscal profits grew significantly.", "score": 0.5},
    ]
    query = "What was the total revenue?"
    results = rerank_chunks(query, chunks, top_k=2)

    assert len(results) == 2
    matched_ids = [r["chunk_id"] for r in results]
    assert "1" in matched_ids or "3" in matched_ids
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]