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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.agents.manager_v2 import run_manager_v2
from app.agents.state import WorkflowState
from app.deps import get_current_user
from app.services.datasets import build_data_summary, get_dataset_dataframe, get_dataset_record
from app.services.domain_router import classify_dataset

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRunRequest(BaseModel):
    query: str
    dataset_id: str


@router.post("/run")
async def run_agent(payload: AgentRunRequest, user=Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
    data_summary = build_data_summary(dataframe)

    # Domain detection still runs, but only as context for the Manager's
    # reasoning now - not as the sole basis for agent selection.
    metadata = get_dataset_record(payload.dataset_id, user.id)
    route = classify_dataset(metadata["filename"], dataframe)

    state = WorkflowState(
        dataset_id=payload.dataset_id,
        user_id=user.id,
        data_summary=data_summary,
        classification=route.to_dict(),
        user_query=payload.query.strip(),
    )

    final_state = await run_manager_v2(state)

    if final_state.final_output is None:
        raise HTTPException(status_code=500, detail="Manager produced no output")

    return {
        "workflow_id": final_state.workflow_id,
        "status": final_state.status,
        "result": final_state.final_output,
    }
