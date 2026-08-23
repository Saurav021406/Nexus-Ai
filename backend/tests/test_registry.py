from app.agents.registry import (
    available_agent_domains,
    get_agent,
    get_agent_definition,
    list_capabilities,
)

EXPECTED_AGENTS = {
    "Education", "Retail", "Finance", "HR", "Healthcare", "General",
    "Data Engineer", "Data Scientist", "ML Engineer", "Business Analyst",
    "Visualization", "SQL", "Research",
}


def test_all_expected_agents_are_registered():
    assert available_agent_domains() == EXPECTED_AGENTS


def test_get_agent_returns_callable_for_known_domain():
    assert callable(get_agent("Finance"))


def test_get_agent_returns_none_for_unknown_domain():
    assert get_agent("NotARealAgent") is None


def test_sql_agent_needs_full_workflow_access():
    definition = get_agent_definition("SQL")
    assert definition is not None
    assert definition.needs_full_access is True


def test_reasoning_only_agents_do_not_need_full_access():
    for name in ["Finance", "Education", "Data Scientist", "Business Analyst"]:
        definition = get_agent_definition(name)
        assert definition.needs_full_access is False, f"{name} should not need full access"


def test_list_capabilities_covers_every_registered_agent():
    caps = list_capabilities()
    assert len(caps) == len(EXPECTED_AGENTS)
    for entry in caps:
        assert "name" in entry
        assert "capabilities" in entry
        assert "tools" in entry
        assert "permissions" in entry


def test_registered_tools_actually_exist_in_tool_registry():
    """The module-load-time fail-fast check in registry.py already
    guarantees this (an unknown tool name would have raised on import,
    which means this test module wouldn't have even loaded) - this test
    documents and re-asserts that guarantee explicitly."""
    from app.agents.tools import list_tools

    known_tool_names = {t["name"] for t in list_tools()}
    for entry in list_capabilities():
        for tool_name in entry["tools"]:
            assert tool_name in known_tool_names, (
                f"Agent '{entry['name']}' references unknown tool '{tool_name}'"
            )
