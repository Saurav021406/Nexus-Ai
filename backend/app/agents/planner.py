"""Task Planner (Section 13 of the Phase 4 spec).

Takes the raw plan the Manager Agent's LLM call produced (agents/manager_v2.py's
AgentPlan) and turns it into something safe and mechanical to execute:

  1. Validates it - unknown agent names, duplicate task ids, depends_on
     pointing at a task id that doesn't exist, and dependency cycles are all
     caught HERE, before anything runs, rather than surfacing as a confusing
     runtime failure mid-execution.
  2. Orders it into "waves" via a topological sort (Kahn's algorithm) - each
     wave is the maximal set of tasks that can run in parallel given the
     tasks in earlier waves have completed. This is what actually
     enforces depends_on, which Steps 1-6 explicitly did not do yet.
  3. Attaches default retry/timeout policy to each task if the plan didn't
     specify one (Section 13: "Retry policy, Timeout" are required fields).

This module has no knowledge of HOW a task executes (that's
agents/executor.py) or of domain specialists specifically - it only
reasons about task ids, agent names, and dependency edges, so it stays
reusable once functional agents beyond the domain specialists exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_RETRIES = 1
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class PlanValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    # Each wave is a list of task dicts that can run concurrently, in order.
    waves: list[list[dict]] = field(default_factory=list)


def _to_task_dicts(tasks) -> list[dict]:
    """Accepts either PlannedTask pydantic models or plain dicts (the
    latter matters for re-planning / resuming from persisted state)."""
    out = []
    for t in tasks:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        d.setdefault("depends_on", [])
        d.setdefault("status", "pending")
        d.setdefault("retries", 0)
        d.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        d.setdefault("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        d.setdefault("result", None)
        d.setdefault("error", None)
        out.append(d)
    return out


def validate_and_order_plan(tasks, available_agents: set[str]) -> PlanValidationResult:
    task_dicts = _to_task_dicts(tasks)
    errors: list[str] = []

    if not task_dicts:
        return PlanValidationResult(valid=False, errors=["Plan has no tasks"], waves=[])

    ids_seen: set[str] = set()
    for t in task_dicts:
        if t["id"] in ids_seen:
            errors.append(f"Duplicate task id: {t['id']}")
        ids_seen.add(t["id"])
        if t["agent"] not in available_agents:
            errors.append(f"Task {t['id']} references unknown agent: {t['agent']}")

    all_ids = {t["id"] for t in task_dicts}
    for t in task_dicts:
        for dep in t["depends_on"]:
            if dep not in all_ids:
                errors.append(f"Task {t['id']} depends_on unknown task id: {dep}")
            if dep == t["id"]:
                errors.append(f"Task {t['id']} depends on itself")

    if errors:
        return PlanValidationResult(valid=False, errors=errors, waves=[])

    # Kahn's algorithm: repeatedly pull out tasks whose dependencies are
    # all already placed in an earlier wave. Whatever's left after no wave
    # could be formed is part of a cycle.
    by_id = {t["id"]: t for t in task_dicts}
    remaining = dict(by_id)
    placed: set[str] = set()
    waves: list[list[dict]] = []

    while remaining:
        wave = [t for t in remaining.values() if all(dep in placed for dep in t["depends_on"])]
        if not wave:
            cycle_ids = list(remaining.keys())
            errors.append(f"Dependency cycle detected among tasks: {cycle_ids}")
            return PlanValidationResult(valid=False, errors=errors, waves=[])

        waves.append(wave)
        for t in wave:
            placed.add(t["id"])
            del remaining[t["id"]]

    return PlanValidationResult(valid=True, errors=[], waves=waves)
