"""Read-only authenticated audit history endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.deps import get_current_user
from app.services import audit_logs

router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


@router.get("")
async def get_audit_logs(
    limit: int = Query(default=100, ge=1, le=200),
    user=Depends(get_current_user),
):
    try:
        events = await run_in_threadpool(audit_logs.list_audit_events, user.id, limit)
    except audit_logs.AuditLogStorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"events": events}
