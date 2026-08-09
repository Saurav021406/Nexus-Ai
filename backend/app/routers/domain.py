"""Dataset classification and collaborative specialist analysis endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.registry import available_agent_domains, get_agent
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
    # Optional by design: the normal path always uses the automatic route. A
    # future UI can offer an accessible human override without changing API
    # semantics; it is strictly restricted to installed specialists.
    selected_domains: list[str] | None = Field(default=None, max_length=3)


def _route_for_dataset(dataset_id: str, user_id: str) -> tuple[DomainRoute, Any]:
    metadata = get_dataset_record(dataset_id, user_id)
    dataframe = get_dataset_dataframe(dataset_id, user_id)
    return classify_dataset(metadata["filename"], dataframe), dataframe


def _normalise_agent_result(domain: str, result: object) -> dict[str, Any]:
    """Keep a malformed model response isolated to that specialist."""
    if not isinstance(result, dict):
        raise ValueError("Specialist returned an invalid response")

    summary = result.get("summary")
    key_metrics = result.get("key_metrics")
    recommendation = result.get("recommendation")
    if not isinstance(summary, str) or not isinstance(key_metrics, list) or not isinstance(recommendation, str):
        raise ValueError("Specialist response did not match the required analysis format")

    return {
        "domain": domain,
        "summary": summary,
        "key_metrics": [str(metric) for metric in key_metrics],
        "recommendation": recommendation,
    }


def _run_specialist(domain: str, data_summary: str) -> dict[str, Any]:
    agent_func = get_agent(domain)
    if not agent_func:
        return {"domain": domain, "error": "This specialist is not installed."}

    try:
        return _normalise_agent_result(domain, agent_func(data_summary))
    except Exception:
        # Do not surface provider internals or prompt data to the client.
        return {
            "domain": domain,
            "error": "This specialist could not complete its analysis. Please try again.",
        }


def _combine_specialist_results(
    route: DomainRoute,
    specialist_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_reports = [report for report in specialist_reports if "error" not in report]
    if not successful_reports:
        raise HTTPException(
            status_code=502,
            detail="No specialist analysis could be completed. Please try again.",
        )

    primary_report = next(
        (report for report in successful_reports if report["domain"] == route.primary_domain),
        successful_reports[0],
    )

    # The orchestrator intentionally does not ask another LLM to restate facts.
    # It presents specialist outputs alongside their source domain, preserving
    # traceability and avoiding a final hallucination-prone summarisation pass.
    key_metrics: list[str] = []
    for report in successful_reports:
        for metric in report["key_metrics"]:
            if metric not in key_metrics:
                key_metrics.append(metric)

    return {
        "classification": route.to_dict(),
        "summary": primary_report["summary"],
        "key_metrics": key_metrics[:8],
        "recommendation": primary_report["recommendation"],
        "participating_agents": [report["domain"] for report in successful_reports],
        "specialist_reports": specialist_reports,
    }


@router.post("/detect")
async def detect_domain(payload: DomainDetectRequest, user=Depends(get_current_user)):
    route, _ = _route_for_dataset(payload.dataset_id, user.id)
    return route.to_dict()


@router.post("/analyze")
async def analyze_domain(payload: AnalyzeRequest, user=Depends(get_current_user)):
    route, dataframe = _route_for_dataset(payload.dataset_id, user.id)
    agent_domains = route.agent_domains

    if payload.selected_domains:
        installed_domains = available_agent_domains()
        unknown_domains = set(payload.selected_domains) - installed_domains
        if unknown_domains:
            raise HTTPException(
                status_code=400,
                detail="One or more selected analysis domains are unavailable.",
            )
        # A user-selected route is an intentional correction, not a permanent
        # training signal. Feedback storage and model evaluation can be added
        # independently once a reviewed-label workflow exists.
        agent_domains = list(dict.fromkeys(payload.selected_domains))

    data_summary = build_data_summary(dataframe)
    specialist_reports = await asyncio.gather(
        *(asyncio.to_thread(_run_specialist, domain, data_summary) for domain in agent_domains)
    )
    return _combine_specialist_results(route, specialist_reports)
