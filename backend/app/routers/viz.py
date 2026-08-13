"""Visualization Agent endpoints."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe
from app.services.visualization import (
    generate_chart,
    explain_chart,
    suggest_multiple_charts_from_request,
    generate_dashboard,
)

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


def _build_one_chart(df: pd.DataFrame, chart_type: str, x, y, color, agg, title) -> dict:
    figure = generate_chart(df, chart_type=chart_type, x=x, y=y, color=color, agg=agg, title=title)
    insight = explain_chart(df, chart_type=chart_type, x=x, y=y, agg=agg)
    return {
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "title": title,
        "figure": figure,
        "insight": insight,
    }


@router.post("/generate")
async def generate(payload: GenerateRequest, user=Depends(get_current_user)):
    """Always returns {"charts": [...]} - a single manually-built chart is a
    1-item list; a natural-language request can return up to 3 charts if the
    request genuinely calls for multiple angles on the data."""
    df = get_dataset_dataframe(payload.dataset_id, user.id)

    # ---------- Natural language path: can produce multiple charts ----------
    if payload.nl_request and not payload.chart_type:
        try:
            suggestions = await run_in_threadpool(
                suggest_multiple_charts_from_request, _columns_info(df), payload.nl_request
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not interpret your request: {e}")

        if not suggestions:
            raise HTTPException(status_code=500, detail="Could not come up with a chart for that request")

        charts = []
        for s in suggestions:
            try:
                chart = await run_in_threadpool(
                    _build_one_chart,
                    df,
                    s.get("chart_type"),
                    s.get("x"),
                    s.get("y"),
                    None,
                    s.get("agg", "sum"),
                    s.get("title"),
                )
                chart["reasoning"] = s.get("reasoning")
                charts.append(chart)
            except Exception:
                continue

        if not charts:
            raise HTTPException(status_code=500, detail="Could not build any of the suggested charts")

        return {"charts": charts, "interpreted": True}

    # ---------- Manual path: exactly one chart ----------
    if not payload.chart_type:
        raise HTTPException(status_code=400, detail="chart_type or nl_request is required")

    try:
        chart = await run_in_threadpool(
            _build_one_chart, df, payload.chart_type, payload.x, payload.y, payload.color, payload.agg, payload.title
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")

    return {"charts": [chart], "interpreted": False}


@router.post("/dashboard")
async def dashboard(payload: DashboardRequest, user=Depends(get_current_user)):
    df = get_dataset_dataframe(payload.dataset_id, user.id)
    charts = await run_in_threadpool(generate_dashboard, df)
    if not charts:
        raise HTTPException(status_code=400, detail="Could not generate any charts for this dataset")
    return {"charts": charts}
