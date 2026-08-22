from app.agents.planner import validate_and_order_plan

AVAILABLE = {"A", "B", "C"}


def test_independent_tasks_land_in_one_wave():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t2", "agent": "B", "depends_on": []},
    ]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert result.valid
    assert len(result.waves) == 1
    assert {t["id"] for t in result.waves[0]} == {"t1", "t2"}


def test_dependent_task_lands_in_a_later_wave():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t2", "agent": "B", "depends_on": []},
        {"id": "t3", "agent": "C", "depends_on": ["t1", "t2"]},
    ]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert result.valid
    assert len(result.waves) == 2
    assert result.waves[1][0]["id"] == "t3"


def test_chain_of_three_produces_three_waves():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t2", "agent": "B", "depends_on": ["t1"]},
        {"id": "t3", "agent": "C", "depends_on": ["t2"]},
    ]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert result.valid
    assert len(result.waves) == 3


def test_direct_cycle_is_rejected():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": ["t2"]},
        {"id": "t2", "agent": "B", "depends_on": ["t1"]},
    ]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert not result.valid
    assert any("cycle" in e.lower() for e in result.errors)


def test_self_dependency_is_rejected():
    tasks = [{"id": "t1", "agent": "A", "depends_on": ["t1"]}]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert not result.valid


def test_unknown_agent_name_is_rejected():
    tasks = [{"id": "t1", "agent": "NotRegistered", "depends_on": []}]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert not result.valid


def test_duplicate_task_id_is_rejected():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t1", "agent": "B", "depends_on": []},
    ]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert not result.valid


def test_dependency_on_nonexistent_task_id_is_rejected():
    tasks = [{"id": "t1", "agent": "A", "depends_on": ["ghost"]}]
    result = validate_and_order_plan(tasks, AVAILABLE)
    assert not result.valid


def test_empty_plan_is_rejected():
    result = validate_and_order_plan([], AVAILABLE)
    assert not result.valid


def test_default_retry_and_timeout_are_attached():
    tasks = [{"id": "t1", "agent": "A", "depends_on": []}]
    result = validate_and_order_plan(tasks, AVAILABLE)
    task = result.waves[0][0]
    assert "max_retries" in task
    assert "timeout_seconds" in task
    assert task["status"] == "pending"
