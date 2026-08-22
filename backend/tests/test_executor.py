import asyncio

from app.agents.executor import execute_plan
from app.agents.planner import validate_and_order_plan
from app.agents.state import WorkflowState


def make_state():
    return WorkflowState(dataset_id="d", user_id="u", data_summary="x", classification={})


async def test_dependency_ordering_is_actually_enforced_at_runtime():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t2", "agent": "B", "depends_on": []},
        {"id": "t3", "agent": "C", "depends_on": ["t1", "t2"]},
    ]
    plan = validate_and_order_plan(tasks, {"A", "B", "C"})

    order = []

    async def runner(task, state):
        order.append(("start", task["id"]))
        await asyncio.sleep(0.01)
        order.append(("end", task["id"]))
        return {"summary": "ok", "key_metrics": [], "recommendation": ""}

    state = make_state()
    await execute_plan(state, plan.waves, runner)

    t3_start = order.index(("start", "t3"))
    assert t3_start > order.index(("end", "t1"))
    assert t3_start > order.index(("end", "t2"))
    assert all(t["status"] == "completed" for t in state.tasks)


async def test_independent_tasks_run_concurrently_not_sequentially():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": []},
        {"id": "t2", "agent": "B", "depends_on": []},
    ]
    plan = validate_and_order_plan(tasks, {"A", "B"})

    concurrent_count = {"max": 0, "current": 0}

    async def runner(task, state):
        concurrent_count["current"] += 1
        concurrent_count["max"] = max(concurrent_count["max"], concurrent_count["current"])
        await asyncio.sleep(0.05)
        concurrent_count["current"] -= 1
        return {"summary": "ok", "key_metrics": [], "recommendation": ""}

    state = make_state()
    await execute_plan(state, plan.waves, runner)
    assert concurrent_count["max"] == 2, "both tasks in one wave should overlap in time"


async def test_failed_task_causes_dependents_to_be_skipped_not_run():
    tasks = [
        {"id": "t1", "agent": "A", "depends_on": [], "max_retries": 0},
        {"id": "t2", "agent": "B", "depends_on": ["t1"]},
        {"id": "t3", "agent": "C", "depends_on": []},
    ]
    plan = validate_and_order_plan(tasks, {"A", "B", "C"})

    calls = {"A": 0, "B": 0, "C": 0}

    async def runner(task, state):
        calls[task["agent"]] += 1
        if task["agent"] == "A":
            return {"error": "boom"}
        return {"summary": "ok", "key_metrics": [], "recommendation": ""}

    state = make_state()
    await execute_plan(state, plan.waves, runner)

    statuses = {t["id"]: t["status"] for t in state.tasks}
    assert statuses["t1"] == "failed"
    assert statuses["t2"] == "skipped"
    assert statuses["t3"] == "completed"
    assert calls["B"] == 0, "B must never actually run once its dependency failed"
    assert len(state.errors) == 1


async def test_transient_failure_recovers_via_retry():
    tasks = [{"id": "t1", "agent": "A", "depends_on": [], "max_retries": 1}]
    plan = validate_and_order_plan(tasks, {"A"})

    calls = {"n": 0}

    async def flaky_runner(task, state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "transient"}
        return {"summary": "recovered", "key_metrics": [], "recommendation": ""}

    state = make_state()
    await execute_plan(state, plan.waves, flaky_runner)
    assert calls["n"] == 2
    assert state.tasks[0]["status"] == "completed"


async def test_persistent_failure_exhausts_retries_and_fails():
    tasks = [{"id": "t1", "agent": "A", "depends_on": [], "max_retries": 2}]
    plan = validate_and_order_plan(tasks, {"A"})

    calls = {"n": 0}

    async def always_fails(task, state):
        calls["n"] += 1
        return {"error": "permanent"}

    state = make_state()
    await execute_plan(state, plan.waves, always_fails)
    assert calls["n"] == 3  # initial attempt + 2 retries
    assert state.tasks[0]["status"] == "failed"


async def test_hung_task_is_cut_off_at_its_timeout():
    tasks = [{"id": "t1", "agent": "A", "depends_on": [], "timeout_seconds": 0.05, "max_retries": 0}]
    plan = validate_and_order_plan(tasks, {"A"})

    async def slow_runner(task, state):
        await asyncio.sleep(5)
        return {"summary": "too slow"}

    state = make_state()
    await execute_plan(state, plan.waves, slow_runner)
    assert state.tasks[0]["status"] == "failed"
    assert "Timed out" in state.tasks[0]["error"]
