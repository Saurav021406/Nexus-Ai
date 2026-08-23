import httpx
import pytest
from fastapi import HTTPException

import app.routers.automation as automation_router
from app.config import settings
from app.services.windmill_client import (
    WindmillNotConfiguredError,
    is_windmill_configured,
    trigger_windmill_workflow,
)


@pytest.fixture(autouse=True)
def _reset_windmill_settings():
    """Every test starts from "not configured" and restores it afterward,
    so tests can't leak Windmill credentials into each other."""
    original = (settings.windmill_base_url, settings.windmill_token, settings.windmill_workspace)
    settings.windmill_base_url = ""
    settings.windmill_token = ""
    settings.windmill_workspace = ""
    yield
    settings.windmill_base_url, settings.windmill_token, settings.windmill_workspace = original


def test_unconfigured_windmill_raises_clear_error_not_a_crash():
    assert is_windmill_configured() is False
    with pytest.raises(WindmillNotConfiguredError):
        trigger_windmill_workflow("u/admin/send_email", {"to": "x@example.com"})


def test_configured_windmill_calls_the_correct_url_and_auth(monkeypatch):
    settings.windmill_base_url = "https://windmill.example.com"
    settings.windmill_token = "fake-token"
    settings.windmill_workspace = "myworkspace"
    assert is_windmill_configured() is True

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "sent"}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    result = trigger_windmill_workflow("u/admin/send_email", {"to": "x@example.com"})
    assert "myworkspace" in captured["url"]
    assert "u/admin/send_email" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer fake-token"
    assert result["success"] is True


class _FakeUser:
    id = "u1"


def _make_approval(status: str) -> dict:
    return {
        "id": "a1",
        "resource_type": "report",
        "resource_id": "r1",
        "content": {"summary": "draft"},
        "approval_status": status,
    }


async def test_pending_approval_is_blocked_and_windmill_is_never_called(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(automation_router, "trigger_windmill_workflow", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(automation_router.approvals, "get_version", lambda approval_id, user_id: _make_approval("pending"))

    payload = automation_router.TriggerRequest(approval_id="a1", script_path="u/admin/send_email")
    with pytest.raises(HTTPException) as exc_info:
        await automation_router.trigger_automation(payload, _FakeUser())

    assert exc_info.value.status_code == 403
    assert calls["n"] == 0, "Windmill must never be contacted for unapproved content"


async def test_rejected_approval_is_also_blocked(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(automation_router, "trigger_windmill_workflow", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(automation_router.approvals, "get_version", lambda approval_id, user_id: _make_approval("rejected"))

    payload = automation_router.TriggerRequest(approval_id="a1", script_path="u/admin/send_email")
    with pytest.raises(HTTPException) as exc_info:
        await automation_router.trigger_automation(payload, _FakeUser())

    assert exc_info.value.status_code == 403
    assert calls["n"] == 0


async def test_approved_content_actually_triggers_windmill(monkeypatch):
    calls = {"n": 0}

    def fake_trigger(script_path, payload):
        calls["n"] += 1
        return {"success": True, "result": {"status": "sent"}}

    monkeypatch.setattr(automation_router, "trigger_windmill_workflow", fake_trigger)
    monkeypatch.setattr(automation_router.approvals, "get_version", lambda approval_id, user_id: _make_approval("approved"))

    payload = automation_router.TriggerRequest(approval_id="a1", script_path="u/admin/send_email")
    result = await automation_router.trigger_automation(payload, _FakeUser())

    assert calls["n"] == 1
    assert result["success"] is True


async def test_missing_approval_returns_404(monkeypatch):
    monkeypatch.setattr(automation_router.approvals, "get_version", lambda approval_id, user_id: None)

    payload = automation_router.TriggerRequest(approval_id="does-not-exist", script_path="u/admin/send_email")
    with pytest.raises(HTTPException) as exc_info:
        await automation_router.trigger_automation(payload, _FakeUser())

    assert exc_info.value.status_code == 404
