from fastapi import Header, HTTPException, status
from app.supabase_client import supabase_anon


async def get_current_user(authorization: str | None = Header(default=None)):
    """
    Expects header: Authorization: Bearer <supabase_access_token>
    The frontend gets this token from supabase.auth.getSession() after login.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        user_response = supabase_anon.auth.get_user(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    if not user_response or not user_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    return user_response.user
