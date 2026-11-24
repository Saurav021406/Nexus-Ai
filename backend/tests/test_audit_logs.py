from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import app.main as main_module
import app.routers.meeting_intelligence as meeting_router
from app.main import app
from app.services import audit_logs


class FakeUser:
    id = "user-1"


def test_record_audit_event_stores_only_metadata(monkeypatch):
    inserted = []

    class FakeTable:
        def insert(self, row):
            inserted.append(row)
            return self

        def execute(self):
            return SimpleNamespace(data=[inserted[-1]])

    class FakeSupabase:
        def table(self, name):
            assert name == audit_logs.TABLE
            return FakeTable()

    monkeypatch.setattr(audit_logs, "supabase_admin", FakeSupabase())
    audit_logs.record_audit_event(
        "user-1",
        "POST /email-generator/generate",
        "/email-generator/generate",
        200,
        125,
    )

    assert inserted == [
        {
            "user_id": "user-1",
            "action": "POST /email-generator/generate",
            "endpoint": "/email-generator/generate",
            "status_code": 200,
            "duration_ms": 125,
        }
    ]


def test_authenticated_ai_action_is_recorded_without_request_content(monkeypatch):
    recorded = []

    async def fake_current_user(request: Request):
        request.state.user_id = FakeUser.id
        return FakeUser()

    def fake_consensus(*_args, **_kwargs):
        return {
            "summary": "Summary",
            "tasks": [],
            "decisions": [],
            "risks": [],
            "open_questions": [],
        }

    monkeypatch.setattr(main_module, "record_audit_event", lambda *args: recorded.append(args))
    monkeypatch.setattr(meeting_router, "get_consensus_json", fake_consensus)
    app.dependency_overrides[meeting_router.get_current_user] = fake_current_user
    try:
        response = TestClient(app).post(
            "/meeting-intelligence/analyze",
            json={
                "transcript": "Private test transcript text",
                "consent_to_ai_processing": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(recorded) == 1
    assert recorded[0][:4] == (
        FakeUser.id,
        "POST /meeting-intelligence/analyze",
        "/meeting-intelligence/analyze",
        200,
    )
