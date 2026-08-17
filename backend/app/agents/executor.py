"""Task Executor (Section 14 of the Phase 4 spec).

Runs the wave-ordered task graph produced by agents/planner.py against
WorkflowState:
  - tasks within a wave run concurrently (asyncio.gather)
  - waves run in order, so a task never starts before its dependencies
    have actually finished - this is the piece Steps 1-6 explicitly left
    unenforced
  - each task gets a timeout (asyncio.wait_for) and a retry budget
  - if a task's dependency failed or was skipped, the task is marked
    "skipped" instead of run - failures don't silently cascade into
    running downstream tasks on missing input
  - every outcome (completed/failed/skipped) is written back into
    state.tasks, state.specialist_results, state.errors, and state.traces

Independent of specialist implementation (Section 14: "The executor must be
independent of individual specialist implementation") - it calls whatever
async `task_runner(task, state)` callable it's given. agents/manager_v2.py
supplies the actual domain-specialist runner; a future SQL/Research/ML
agent just needs its own runner function with the same signature.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from app.agents.state import WorkflowState
from app.agents.tracing import add_trace

TaskRunner = Callable[[dict, WorkflowState], Awaitable[dict]]


async def _run_with_retry(task: dict, state: WorkflowState, task_runner: TaskRunner) -> None:
    task["status"] = "running"
    max_retries = task.get("max_retries", 1)
    timeout = task.get("timeout_seconds", 60)

    attempt = 0
    last_error: str | None = None

    while attempt <= max_retries:
        try:
            output = await asyncio.wait_for(task_runner(task, state), timeout=timeout)
            if not isinstance(output, dict):
                output = {"error": "Task runner returned a non-dict result"}
            if "error" in output:
                last_error = output["error"]
                raise RuntimeError(last_error)

            task["status"] = "completed"
            task["result"] = output
            task["retries"] = attempt
            add_trace(state, task["agent"], "completed", output_data=output)
            state.specialist_results[task["agent"]] = output
            return

        except asyncio.TimeoutError:
            last_error = f"Timed out after {timeout}s"
        except Exception as e:
            last_error = str(e)

        attempt += 1
        if attempt <= max_retries:
            add_trace(
                state, task["agent"], "retrying",
                error=f"attempt {attempt}/{max_retries}: {last_error}",
            )

    # All attempts exhausted.
    task["status"] = "failed"
    task["error"] = last_error
    task["retries"] = max_retries
    error_result = {"error": last_error or "Task failed"}
    task["result"] = error_result
    state.specialist_results[task["agent"]] = error_result
    state.add_error(task["agent"], last_error or "Task failed")
    add_trace(state, task["agent"], "failed", error=last_error)


def _mark_skipped(task: dict, state: WorkflowState, reason: str) -> None:
    task["status"] = "skipped"
    task["error"] = reason
    add_trace(state, task["agent"], "skipped", error=reason)


async def execute_plan(
    state: WorkflowState,
    waves: list[list[dict]],
    task_runner: TaskRunner,
) -> None:
    """Mutates state in place: state.tasks ends up with per-task
    status/result/error, state.specialist_results gets each successful
    task's output keyed by agent name (same shape the original manager.py
    always produced, so _synthesize() in manager_v2.py needs no changes),
    and state.errors collects every failure without stopping the rest of
    the workflow.
    """
    state.tasks = [t for wave in waves for t in wave]
    failed_or_skipped_ids: set[str] = set()

    for wave in waves:
        runnable: list[dict] = []
        for task in wave:
            blocking = [dep for dep in task["depends_on"] if dep in failed_or_skipped_ids]
            if blocking:
                _mark_skipped(
                    task, state, f"Skipped - dependency failed/skipped: {', '.join(blocking)}"
                )
                failed_or_skipped_ids.add(task["id"])
            else:
                runnable.append(task)

        if runnable:
            await asyncio.gather(*(_run_with_retry(t, state, task_runner) for t in runnable))
            for t in runnable:
                if t["status"] in ("failed", "skipped"):
                    failed_or_skipped_ids.add(t["id"])
