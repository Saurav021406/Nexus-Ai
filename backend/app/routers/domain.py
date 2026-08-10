"""Dataset classification and multi-agent analysis endpoints (Phase 4)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.manager import run_manager
from app.agents.registry import available_agent_domains
from app.agents.state import WorkflowState
from app.deps import get_current_user
from app.services.datasets import (
    build_data_summary,
    get_dataset_dataframe,
    get_dataset_record,
)
from app.services.domain_router import DomainRoute, classify_dataset

router = APIRouter(prefix="/domain", tags=["domain"])


class DomainDetectRequest(BaseModel):
    dataset_id: str


class AnalyzeRequest(BaseModel):
    dataset_id: str
    selected_domains: list[str] | None = Field(default=None, max_length=3)


def _route_for_dataset(dataset_id: str, user_id: str) -> tuple[DomainRoute, Any]:
    metadata = get_dataset_record(dataset_id, user_id)
    dataframe = get_dataset_dataframe(dataset_id, user_id)
    return classify_dataset(metadata["filename"], dataframe), dataframe


@router.post("/detect")
async def detect_domain(payload: DomainDetectRequest, user=Depends(get_current_user)):
    route, _ = _route_for_dataset(payload.dataset_id, user.id)
    return route.to_dict()


@router.post("/analyze")
async def analyze_domain(payload: AnalyzeRequest, user=Depends(get_current_user)):
    route, dataframe = _route_for_dataset(payload.dataset_id, user.id)

    # Allow optional human override of domains
    if payload.selected_domains:
        installed = available_agent_domains()
        unknown = set(payload.selected_domains) - installed
        if unknown:
            raise HTTPException(
                status_code=400,
                detail="One or more selected analysis domains are unavailable.",
            )
        # Force the selected domains into the route for this run
        route = DomainRoute(
            primary_domain=payload.selected_domains[0],
            secondary_domains=payload.selected_domains[1:],
            tags=route.tags,
            confidence=route.confidence,
            reasoning="User-selected domains",
            agent_domains=payload.selected_domains,
            candidates=route.candidates,
        )

    data_summary = build_data_summary(dataframe)

    state = WorkflowState(
        dataset_id=payload.dataset_id,
        user_id=user.id,
        data_summary=data_summary,
        classification=route.to_dict(),
    )

    final_state = await run_manager(state)

    if final_state.final_output is None:
        raise HTTPException(status_code=500, detail="Manager produced no output")

    return final_state.final_output
