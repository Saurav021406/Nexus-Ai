"""Exploratory Data Analysis endpoints: correlation matrix and per-column
distributions, used to power simple charts on the frontend."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe

router = APIRouter(prefix="/eda", tags=["eda"])

MAX_CORRELATION_COLUMNS = 15
HISTOGRAM_BINS = 10
MAX_CATEGORIES = 10


class DatasetIdRequest(BaseModel):
    dataset_id: str


class DistributionRequest(BaseModel):
    dataset_id: str
    column: str


@router.post("/correlation")
async def get_correlation(payload: DatasetIdRequest, user=Depends(get_current_user)):
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 numeric columns to compute a correlation matrix.",
        )

    # Cap column count so the matrix stays readable in the UI
    numeric_df = numeric_df.iloc[:, :MAX_CORRELATION_COLUMNS]

    corr = numeric_df.corr(numeric_only=True).round(2)
    corr = corr.fillna(0)

    return {
        "columns": corr.columns.tolist(),
        "matrix": corr.values.tolist(),
    }


@router.post("/columns")
async def get_chartable_columns(payload: DatasetIdRequest, user=Depends(get_current_user)):
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique() <= 50
    ]
    return {"numeric_columns": numeric_cols, "categorical_columns": categorical_cols}


@router.post("/distribution")
async def get_distribution(payload: DistributionRequest, user=Depends(get_current_user)):
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)

    if payload.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{payload.column}' not found")

    series = df[payload.column].dropna()
    if len(series) == 0:
        raise HTTPException(status_code=400, detail="This column has no non-empty values")

    if pd.api.types.is_numeric_dtype(series):
        counts, bin_edges = pd.cut(series, bins=HISTOGRAM_BINS, retbins=True, duplicates="drop")
        value_counts = counts.value_counts().sort_index()
        bars = [
            {"label": f"{round(interval.left, 1)} - {round(interval.right, 1)}", "value": int(count)}
            for interval, count in value_counts.items()
        ]
        return {"type": "histogram", "column": payload.column, "bars": bars}

    value_counts = series.value_counts().head(MAX_CATEGORIES)
    bars = [{"label": str(label), "value": int(count)} for label, count in value_counts.items()]
    return {"type": "category", "column": payload.column, "bars": bars}
