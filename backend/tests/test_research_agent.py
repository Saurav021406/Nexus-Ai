import app.agents.research_agent as research_agent


def test_research_agent_always_flags_itself_as_ungrounded(monkeypatch):
    """Section 16: full RAG belongs to Phase 6 - until then, every answer
    from this agent must be explicitly marked as not backed by any
    retrieved source, so callers/UI never mistake it for grounded output."""
    monkeypatch.setattr(
        research_agent, "get_consensus_json",
        lambda prompt, **kw: {"summary": "some general knowledge answer", "key_metrics": [], "recommendation": ""},
    )
    result = research_agent.analyze("mean churn: 8%", "What is a typical SaaS churn benchmark?")
    assert result["grounded"] is False


def test_research_agent_falls_back_to_generic_task_description(monkeypatch):
    captured = {}

    def fake_consensus(prompt, **kw):
        captured["prompt"] = prompt
        return {"summary": "x", "key_metrics": [], "recommendation": ""}

    monkeypatch.setattr(research_agent, "get_consensus_json", fake_consensus)
    research_agent.analyze("some data summary", task_description="")
    assert "relevant external context" in captured["prompt"].lower()
