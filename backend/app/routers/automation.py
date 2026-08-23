"""External automation endpoint (Section 33 example: "Approved report ->
Windmill -> Email/Slack/Teams/CRM").

This is the ONLY place in the codebase that calls
services/windmill_client.py. It enforces the precondition Section 7 and 33
both describe: Windmill only ever fires on content a human has already
approved through the generic Human Approval system (services/approvals.py)
- never as something an agent decides to do mid-reasoning-loop, and never
on unreviewed content. If the referenced approval isn't in "approved"
status, this rejects before Windmill is even contacted.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_current_user
from app.services import approvals
from app.services.windmill_client import (
    WindmillNotConfiguredError,
    is_windmill_configured,
    trigger_windmill_workflow,
)

router = APIRouter(prefix="/automation", tags=["automation"])


class TriggerRequest(BaseModel):
    approval_id: str
    script_path: str
    extra_payload: dict = {}


@router.get("/status")
async def automation_status(user=Depends(get_current_user)):
    """Lets the frontend show/hide automation options without guessing -
    no point offering a 'Send via Windmill' button if nothing's configured."""
    return {"windmill_configured": is_windmill_configured()}


@router.post("/trigger")
async def trigger_automation(payload: TriggerRequest, user=Depends(get_current_user)):
    approval = approvals.get_version(payload.approval_id, user.id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    if approval["approval_status"] != "approved":
        raise HTTPException(
            status_code=403,
            detail=(
                f"This content is '{approval['approval_status']}', not approved. "
                "Only approved content can be sent to external automation."
            ),
        )

    windmill_payload = {
        "resource_type": approval["resource_type"],
        "resource_id": approval["resource_id"],
        "content": approval["content"],
        **payload.extra_payload,
    }

    try:
        result = trigger_windmill_workflow(payload.script_path, windmill_payload)
    except WindmillNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"Windmill rejected the request: {e.response.status_code} {e.response.text[:200]}"
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Windmill: {e}")

    return result
