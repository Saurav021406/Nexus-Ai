"""Root Cause Analysis endpoint (one of the 3 starred features)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.deps import get_current_user
from app.services import root_cause
from app.services.datasets import get_dataset_dataframe

router = APIRouter(prefix="/root-cause", tags=["root-cause"])


class RootCauseRequest(BaseModel):
    dataset_id: str
    metric_column: str
    date_column: str
    aggregation: str = "sum"  # "sum" | "mean"
    direction: str = "decrease"  # "decrease" | "increase"
    segment_columns: list[str] | None = None  # None = auto-detect


@router.post("/analyze")
async def analyze_root_cause(payload: RootCauseRequest, user=Depends(get_current_user)):
    def _run() -> dict:
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        try:
            result = root_cause.analyze_root_cause(
                dataframe,
                metric_column=payload.metric_column,
                date_column=payload.date_column,
                aggregation=payload.aggregation,
                direction=payload.direction,
                segment_columns=payload.segment_columns,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        business_summary = root_cause.explain_root_cause(result)

        return {
            "metric_column": result.metric_column,
            "date_column": result.date_column,
            "aggregation": result.aggregation,
            "direction": result.direction,
            "period_before": result.period_before,
            "period_after": result.period_after,
            "overall_before": result.overall_before,
            "overall_after": result.overall_after,
            "overall_change": result.overall_change,
            "overall_pct_change": result.overall_pct_change,
            "segment_columns_analyzed": result.segment_columns_analyzed,
            "top_contributors": [
                {
                    "segment_column": c.segment_column,
                    "segment_value": c.segment_value,
                    "before": c.before,
                    "after": c.after,
                    "delta": c.delta,
                    "contribution_pct": c.contribution_pct,
                }
                for c in result.top_contributors
            ],
            "warnings": result.warnings,
            "business_summary": business_summary,
        }

    return await run_in_threadpool(_run)
