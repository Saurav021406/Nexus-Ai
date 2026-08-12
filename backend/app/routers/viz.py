"""Visualization Agent endpoints."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe
from app.services.visualization import generate_chart, suggest_chart_from_request, generate_dashboard

router = APIRouter(prefix="/viz", tags=["visualization"])


class GenerateRequest(BaseModel):
    dataset_id: str
    chart_type: str | None = None
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str = "sum"
    title: str | None = None
    nl_request: str | None = None


class DashboardRequest(BaseModel):
    dataset_id: str


def _columns_info(df: pd.DataFrame) -> list[dict]:
    info = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text"
        info.append({"name": col, "type": dtype})
    return info


@router.post("/generate")
async def generate(payload: GenerateRequest, user=Depends(get_current_user)):
    df = get_dataset_dataframe(payload.dataset_id, user.id)

    chart_type = payload.chart_type
    x, y, agg, title = payload.x, payload.y, payload.agg, payload.title
    interpreted = False
    reasoning = None

    if payload.nl_request and not chart_type:
        try:
            suggestion = await run_in_threadpool(
                suggest_chart_from_request, _columns_info(df), payload.nl_request
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not interpret your request: {e}")

        chart_type = suggestion.get("chart_type")
        x = suggestion.get("x") or x
        y = suggestion.get("y") or y
        agg = suggestion.get("agg", agg)
        title = suggestion.get("title", title)
        reasoning = suggestion.get("reasoning")
        interpreted = True

    if not chart_type:
        raise HTTPException(status_code=400, detail="chart_type or nl_request is required")

    try:
        figure = await run_in_threadpool(
            generate_chart, df, chart_type, x, y, payload.color, agg, title
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")

    return {
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "title": title,
        "figure": figure,
        "interpreted": interpreted,
        "reasoning": reasoning,
    }


@router.post("/dashboard")
async def dashboard(payload: DashboardRequest, user=Depends(get_current_user)):
    df = get_dataset_dataframe(payload.dataset_id, user.id)
    charts = await run_in_threadpool(generate_dashboard, df)
    if not charts:
        raise HTTPException(status_code=400, detail="Could not generate any charts for this dataset")
    return {"charts": charts}
