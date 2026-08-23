"""Windmill integration (Section 7 of the Phase 4 spec).

    Nexus Intelligence Core
            |
    Decision / Approved Action
            |
        Windmill
            |
    External System (email, Slack, CRM, scheduled jobs, ...)

"Windmill is NOT the Manager Agent. Windmill is NOT the brain of Nexus...
Do not make Windmill responsible for the core agent reasoning loop."

This module is deliberately a thin, dumb HTTP client - it has no reasoning,
no agent logic, nothing decides anything here. It just triggers a Windmill
script/flow by path and returns whatever Windmill returns. All the
"should this actually happen" decision-making lives elsewhere (see
routers/automation.py, which is the only caller and enforces that Windmill
only ever fires on content that's already been through Human Approval).

Nothing in the Tool Registry (agents/tools.py) grants any agent direct
access to this - an LLM agent mid-reasoning-loop can never trigger an
external action on its own. That's intentional: Section 33's example
("Approved report -> Windmill -> Email/Slack/Teams/CRM") only ever fires
Windmill AFTER a human has approved the content, never as a step an agent
decides to take by itself.

Uses httpx directly (already a transitive dependency via openai/supabase -
no new package needed) rather than the openai-compatible client wrapper
used elsewhere, since this isn't talking to an LLM.
"""

from __future__ import annotations

import httpx

from app.config import settings


class WindmillNotConfiguredError(Exception):
    """Raised when WINDMILL_BASE_URL/WINDMILL_TOKEN/WINDMILL_WORKSPACE
    aren't set - callers should treat this as "automation isn't set up
    yet", not as a bug."""


def is_windmill_configured() -> bool:
    return bool(settings.windmill_base_url and settings.windmill_token and settings.windmill_workspace)


def trigger_windmill_workflow(script_path: str, payload: dict) -> dict:
    """Triggers a Windmill script/flow by its workspace path (e.g.
    "u/admin/send_report_email") with the given JSON payload, and waits
    for the result. Uses Windmill's synchronous "run and wait for result"
    endpoint so the caller gets a real success/failure back rather than
    just a job id to poll separately - simpler for Phase 4's scope.

    Raises WindmillNotConfiguredError if no Windmill instance is set up.
    Raises httpx.HTTPStatusError if Windmill itself rejects the request
    (bad script path, bad payload, script-side error, etc.) - callers
    should catch and surface this, not let it crash silently.
    """
    if not is_windmill_configured():
        raise WindmillNotConfiguredError(
            "Windmill is not configured - set WINDMILL_BASE_URL, WINDMILL_TOKEN, "
            "and WINDMILL_WORKSPACE to enable external automation."
        )

    url = (
        f"{settings.windmill_base_url.rstrip('/')}/api/w/{settings.windmill_workspace}"
        f"/jobs/run_wait_result/p/{script_path.lstrip('/')}"
    )
    response = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {settings.windmill_token}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return {"success": True, "result": response.json()}
