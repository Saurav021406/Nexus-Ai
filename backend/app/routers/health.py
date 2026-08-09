from fastapi import APIRouter, Depends
from app.deps import get_current_user

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Public - no auth. Confirms the API is up."""
    return {"status": "ok", "service": "nexus-ai backend"}


@router.get("/me")
async def me(user=Depends(get_current_user)):
    """Protected - confirms Supabase auth is wired correctly end-to-end."""
    return {"id": user.id, "email": user.email}
