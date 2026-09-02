"""Token/cost savings dashboard endpoint.

App-wide, not per-user or per-dataset (see services/usage_stats.py for
why) - still requires login, just doesn't filter by who's asking.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_current_user
from app.services import usage_stats

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/usage")
async def get_usage_stats(user=Depends(get_current_user)):
    return usage_stats.get_stats()
