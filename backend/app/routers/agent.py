"""Intent-aware agent endpoint (Section 31 of the Phase 4 spec).

This is additive: /domain/analyze (routers/domain.py) and the existing "AI
analysis" tab are completely untouched and keep working exactly as before.
/agent/run is the new entry point for the Manager Agent v2, which reasons
over an actual free-text user_query instead of pure domain routing.

Currently synchronous - it runs the full workflow and returns the final
result in one request/response (plus streams live progress via
/agent/run/stream, see below). Every completed run is persisted to Supabase
(services/workflow_runs.py, Section 28) and gets a pending Human Approval
entry (services/approvals.py) - GET /agent/runs and GET /agent/runs/{id}
let a past run be looked back up later, per Section 31.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.events import make_event
from app.agents.manager_v2 import run_manager_v2
from app.agents.state import WorkflowState
from app.deps import get_current_user
from app.services import approvals
from app.services.datasets import build_data_summary, get_dataset_dataframe, get_dataset_record
from app.services.domain_router import classify_dataset
from app.services.workflow_runs import get_workflow_run, list_workflow_runs, save_workflow_run

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRunRequest(BaseModel):
    query: str
    dataset_id: str


def _build_initial_state(payload: AgentRunRequest, user_id: str) -> WorkflowState:
    dataframe = get_dataset_dataframe(payload.dataset_id, user_id)
    data_summary = build_data_summary(dataframe)

    # Domain detection still runs, but only as context for the Manager's
    # reasoning now - not as the sole basis for agent selection.
    metadata = get_dataset_record(payload.dataset_id, user_id)
    route = classify_dataset(metadata["filename"], dataframe)

    return WorkflowState(
        dataset_id=payload.dataset_id,
        user_id=user_id,
        data_summary=data_summary,
        classification=route.to_dict(),
        user_query=payload.query.strip(),
    )


def _create_pending_approval(final_state: WorkflowState, user_id: str) -> dict | None:
    """Generic Human Approval hook (Phase 4 roadmap item): every completed
    Multi-Agent run gets a pending approval row, the same way report
    generation already does for reports (see services/report_versions.py).
    Never blocks/fails the main response - approval is a nice-to-have on
    top of a result the user can already see."""
    if final_state.final_output is None:
        return None
    try:
        return approvals.create_version(
            resource_type="agent_workflow",
            resource_id=final_state.workflow_id,
            user_id=user_id,
            content=final_state.final_output,
            dataset_id=final_state.dataset_id,
        )
    except Exception as e:
        print(f"Could not create pending approval for workflow {final_state.workflow_id}: {e}")
        return None


def _persist_workflow_run(final_state: WorkflowState) -> None:
    """Workflow persistence (Section 28). Same non-blocking pattern as
    _create_pending_approval above - a workflow that ran and answered
    correctly should never come back as an error to the user just because
    saving its history entry failed."""
    try:
        save_workflow_run(final_state)
    except Exception as e:
        print(f"Could not persist workflow run {final_state.workflow_id}: {e}")


@router.get("/runs")
async def get_agent_runs(dataset_id: str | None = None, limit: int = 20, user=Depends(get_current_user)):
    runs = await run_in_threadpool(list_workflow_runs, user.id, dataset_id, limit)
    return {"runs": runs}


@router.get("/runs/{workflow_id}")
async def get_agent_run(workflow_id: str, user=Depends(get_current_user)):
    run = await run_in_threadpool(get_workflow_run, workflow_id, user.id)
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return run


@router.post("/run")
async def run_agent(payload: AgentRunRequest, user=Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    state = await run_in_threadpool(_build_initial_state, payload, user.id)
    final_state = await run_manager_v2(state)

    if final_state.final_output is None:
        raise HTTPException(status_code=500, detail="Manager produced no output")

    approval = _create_pending_approval(final_state, user.id)
    await run_in_threadpool(_persist_workflow_run, final_state)

    return {
        "workflow_id": final_state.workflow_id,
        "status": final_state.status,
        "result": final_state.final_output,
        "approval": approval,
    }


@router.post("/run/stream")
async def run_agent_stream(payload: AgentRunRequest, user=Depends(get_current_user)):
    """Section 30: live workflow progress. Same work as /agent/run, but
    streams a Server-Sent-Events line per stage as it actually happens
    (Manager planning, each task starting/completing/failing/retrying/
    skipping per wave, quality check, final synthesis) instead of making
    the frontend wait for one big response at the end.

    Uses a plain fetch()-readable text/event-stream rather than requiring
    the browser's EventSource API, since EventSource can't send a POST
    body and we need query + dataset_id in the request.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    state = await run_in_threadpool(_build_initial_state, payload, user.id)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event_type: str, agent: str, message: str, data: dict | None = None) -> None:
        await queue.put(make_event(event_type, agent, message, data))

    async def run_and_finish() -> None:
        try:
            final_state = await run_manager_v2(state, emit)
            approval = _create_pending_approval(final_state, state.user_id)
            await run_in_threadpool(_persist_workflow_run, final_state)
            await queue.put({
                "type": "__done__",
                "workflow_id": final_state.workflow_id,
                "status": final_state.status,
                "result": final_state.final_output,
                "approval": approval,
            })
        except Exception as e:
            await queue.put({"type": "__error__", "message": str(e)})

    async def event_stream():
        task = asyncio.create_task(run_and_finish())
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in ("__done__", "__error__"):
                    break
        finally:
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )