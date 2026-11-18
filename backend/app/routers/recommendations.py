"""Dataset-based product recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe
from app.services.recommendations import recommend_products

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationRequest(BaseModel):
    dataset_id: str
    user_column: str
    user_value: str = Field(min_length=1, max_length=500)
    item_column: str
    rating_column: str | None = None
    limit: int = Field(default=10, ge=1, le=20)


@router.post("/generate")
async def generate_recommendations(payload: RecommendationRequest, user=Depends(get_current_user)):
    def _run() -> dict:
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        try:
            return recommend_products(
                dataframe,
                user_column=payload.user_column,
                user_value=payload.user_value,
                item_column=payload.item_column,
                rating_column=payload.rating_column,
                limit=payload.limit,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return await run_in_threadpool(_run)
