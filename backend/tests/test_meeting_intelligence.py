import pytest
from fastapi.testclient import TestClient

import app.routers.meeting_intelligence as meeting_router
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[meeting_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def test_analyze_meeting_requires_sharing_consent(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("AI provider should not be called without consent")

    monkeypatch.setattr(meeting_router, "get_consensus_json", fail_if_called)
    response = TestClient(app).post(
        "/meeting-intelligence/analyze",
        json={"transcript": "Discuss delivery", "consent_to_ai_processing": False},
    )

    assert response.status_code == 400
    assert "permission" in response.json()["detail"]


def test_analyze_meeting_returns_structured_notes(monkeypatch):
    def fake_consensus(prompt, **kwargs):
        assert "Discuss launch next week" in prompt
        assert kwargs["tier"] == "fast"
        return {
            "summary": "The team discussed the launch.",
            "tasks": [
                {"task": "Confirm launch date", "owner": "Alex", "deadline": "Friday", "priority": "high"}
            ],
            "decisions": ["Proceed with launch planning."],
            "risks": [],
            "open_questions": ["Is support coverage ready?"],
            "_consensus": {"models_used": ["groq"]},
        }

    monkeypatch.setattr(meeting_router, "get_consensus_json", fake_consensus)
    response = TestClient(app).post(
        "/meeting-intelligence/analyze",
        json={
            "transcript": "Discuss launch next week",
            "consent_to_ai_processing": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["tasks"][0]["owner"] == "Alex"
    assert response.json()["open_questions"] == ["Is support coverage ready?"]
    assert "_consensus" not in response.json()


def test_analyze_meeting_rejects_transcripts_over_limit():
    response = TestClient(app).post(
        "/meeting-intelligence/analyze",
        json={
            "transcript": "x" * (meeting_router.MAX_TRANSCRIPT_LENGTH + 1),
            "consent_to_ai_processing": True,
        },
    )

    assert response.status_code == 422
