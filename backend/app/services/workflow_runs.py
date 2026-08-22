"""Workflow run persistence (Phase 4 roadmap: "Workflow persistence to
Supabase - save every /agent/run", Section 28).

Every /agent/run and /agent/run/stream call ends by saving the full
WorkflowState snapshot here. Requires the `workflow_runs` table - see
backend/workflow_runs_table.sql for the migration to run in Supabase.

Persistence failures never fail the user-facing request - see routers/
agent.py, which wraps save_workflow_run() in a try/except. A workflow that
ran successfully but failed to save is still a successful workflow from
the user's point of view; losing the history entry is a lesser problem
than losing the answer they were waiting for.
"""

from __future__ import annotations

from typing import Any

from app.agents.state import WorkflowState
from app.supabase_client import supabase_admin

TABLE = "workflow_runs"


def save_workflow_run(state: WorkflowState) -> dict[str, Any]:
    row = {
        "workflow_id": state.workflow_id,
        "user_id": state.user_id,
        "dataset_id": state.dataset_id,
        "user_query": state.user_query,
        "goal": state.goal,
        "status": state.status,
        "state": state.to_dict(),
    }
    result = supabase_admin.table(TABLE).upsert(row, on_conflict="workflow_id").execute()
    if not result.data:
        raise RuntimeError("Failed to save workflow run")
    return result.data[0]


def get_workflow_run(workflow_id: str, user_id: str) -> dict[str, Any] | None:
    result = (
        supabase_admin.table(TABLE)
        .select("*")
        .eq("workflow_id", workflow_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return result.data


def list_workflow_runs(
    user_id: str, dataset_id: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    query = (
        supabase_admin.table(TABLE)
        # Summary columns only - the full `state` blob is fetched via
        # get_workflow_run() when someone actually opens one, not in a list.
        .select("workflow_id, dataset_id, user_query, goal, status, created_at, updated_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if dataset_id:
        query = query.eq("dataset_id", dataset_id)
    result = query.execute()
    return result.data or []
