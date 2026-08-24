"""Time-series / trend forecasting using a real trained scikit-learn model.

Deliberately kept simple and explainable: linear regression over row order
(or a detected date column), reported with an R^2 score so the user can see
how reliable the trend fit is. No LLM involved here - these are Python-computed
predictions, unlike the Consensus-Engine-based analysis/chat features.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe

router = APIRouter(prefix="/forecast", tags=["forecast"])

MAX_FUTURE_PERIODS = 12


class ForecastRequest(BaseModel):
    dataset_id: str
    target_column: str
    periods: int = 5


class ColumnsRequest(BaseModel):
    dataset_id: str


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Best-effort detection of a usable date/time column, if one exists."""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # format="mixed" tells pandas explicitly "yes, formats may
                # vary row to row, parse flexibly" - same fallback behavior
                # as before, but without the UserWarning spamming logs once
                # per column tried (this runs across every object-dtype
                # column, every time /forecast/columns is called).
                parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
                # Consider it a date column if most values parsed successfully
                if parsed.notna().mean() > 0.9:
                    return col
            except Exception:
                continue
    return None


@router.post("/columns")
async def get_forecastable_columns(payload: ColumnsRequest, user=Depends(get_current_user)):
    """Returns which numeric columns are suitable targets for forecasting."""
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        raise HTTPException(
            status_code=400,
            detail="This dataset has no numeric columns, so forecasting isn't available.",
        )

    date_col = _find_date_column(df)
    return {"numeric_columns": numeric_cols, "date_column": date_col}


@router.post("")
async def run_forecast(payload: ForecastRequest, user=Depends(get_current_user)):
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)

    if payload.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{payload.target_column}' not found")

    if not pd.api.types.is_numeric_dtype(df[payload.target_column]):
        raise HTTPException(
            status_code=400, detail=f"Column '{payload.target_column}' is not numeric"
        )

    periods = max(1, min(payload.periods, MAX_FUTURE_PERIODS))

    working = df[[payload.target_column]].dropna().reset_index(drop=True)
    if len(working) < 5:
        raise HTTPException(
            status_code=400,
            detail="Need at least 5 non-empty rows in this column to train a forecast model.",
        )

    date_col = _find_date_column(df)
    x_label = "time period"
    if date_col:
        dated = df[[date_col, payload.target_column]].dropna().reset_index(drop=True)
        dated[date_col] = pd.to_datetime(dated[date_col], format="mixed", errors="coerce")
        dated = dated.dropna().sort_values(date_col).reset_index(drop=True)
        if len(dated) >= 5:
            working = dated
            x_label = date_col

    # X = row order (0, 1, 2, ...) as the time axis. Kept intentionally simple
    # and explainable - a linear trend line, not a black-box model.
    X = np.arange(len(working)).reshape(-1, 1)
    y = working[payload.target_column].values

    model = LinearRegression()
    model.fit(X, y)

    predictions_on_known = model.predict(X)
    r2 = round(float(r2_score(y, predictions_on_known)), 3)
    mae = round(float(mean_absolute_error(y, predictions_on_known)), 2)

    future_X = np.arange(len(working), len(working) + periods).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    slope = float(model.coef_[0])
    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

    history = [
        {"period": i, "actual": round(float(val), 2)}
        for i, val in enumerate(y)
    ]
    forecast = [
        {"period": len(working) + i, "predicted": round(float(val), 2)}
        for i, val in enumerate(future_predictions)
    ]

    return {
        "target_column": payload.target_column,
        "x_axis_label": x_label,
        "trend": trend,
        "slope_per_period": round(slope, 4),
        "r2_score": r2,
        "mean_absolute_error": mae,
        "history": history,
        "forecast": forecast,
        "note": (
            "R^2 close to 1.0 means the linear trend fits the data well. "
            "Values well below that mean the data is noisy or non-linear, "
            "and this forecast should be treated as a rough trend estimate."
        ),
    }
