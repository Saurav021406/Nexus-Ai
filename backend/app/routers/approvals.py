"""Generic Human Approval endpoints (Phase 4 roadmap: "Generic Human
Approval, currently only works for reports").

Works for any resource_type - currently "agent_workflow" (Multi-Agent
results, auto-created by routers/agent.py after a run completes).
Reports keep their own dedicated /report/* endpoints (routers/report.py)
completely unchanged; this is additive, not a replacement.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_current_user
from app.services import approvals

router = APIRouter(prefix="/approvals", tags=["approvals"])


class CreateApprovalRequest(BaseModel):
    resource_type: str
    resource_id: str
    content: dict[str, Any]
    dataset_id: str | None = None


class RejectApprovalRequest(BaseModel):
    reason: str


@router.post("")
async def create_approval(payload: CreateApprovalRequest, user=Depends(get_current_user)):
    try:
        return approvals.create_version(
            payload.resource_type, payload.resource_id, user.id, payload.content, payload.dataset_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save approval: {e}")


@router.get("/{resource_type}/{resource_id}")
async def list_approvals(resource_type: str, resource_id: str, user=Depends(get_current_user)):
    return {"versions": approvals.list_versions(resource_type, resource_id, user.id)}


@router.post("/{approval_id}/approve")
async def approve(approval_id: str, user=Depends(get_current_user)):
    try:
        return approvals.set_status(approval_id, user.id, "approved")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Approval not found: {e}")


@router.post("/{approval_id}/reject")
async def reject(approval_id: str, payload: RejectApprovalRequest, user=Depends(get_current_user)):
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="A rejection reason is required")
    try:
        return approvals.set_status(approval_id, user.id, "rejected", payload.reason.strip())
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Approval not found: {e}")
