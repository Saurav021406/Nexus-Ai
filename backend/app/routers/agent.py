"""Intent-aware agent endpoint (Section 31 of the Phase 4 spec).

This is additive: /domain/analyze (routers/domain.py) and the existing "AI
analysis" tab are completely untouched and keep working exactly as before.
/agent/run is the new entry point for the Manager Agent v2, which reasons
over an actual free-text user_query instead of pure domain routing.

Currently synchronous - it runs the full workflow and returns the final
result in one request/response, same as /domain/analyze does today. Async
polling (GET /agent/runs/{workflow_id}) and persistence to Supabase are
later steps (Section 28/31) once the Task Planner/Executor exist and
workflows can genuinely run long enough to need polling.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.events import make_event
from app.agents.manager_v2 import run_manager_v2
from app.agents.state import WorkflowState
from app.deps import get_current_user
from app.services.datasets import build_data_summary, get_dataset_dataframe, get_dataset_record
from app.services.domain_router import classify_dataset

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


@router.post("/run")
async def run_agent(payload: AgentRunRequest, user=Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    state = _build_initial_state(payload, user.id)
    final_state = await run_manager_v2(state)

    if final_state.final_output is None:
        raise HTTPException(status_code=500, detail="Manager produced no output")

    return {
        "workflow_id": final_state.workflow_id,
        "status": final_state.status,
        "result": final_state.final_output,
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

    state = _build_initial_state(payload, user.id)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event_type: str, agent: str, message: str, data: dict | None = None) -> None:
        await queue.put(make_event(event_type, agent, message, data))

    async def run_and_finish() -> None:
        try:
            final_state = await run_manager_v2(state, emit)
            await queue.put({
                "type": "__done__",
                "workflow_id": final_state.workflow_id,
                "status": final_state.status,
                "result": final_state.final_output,
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
