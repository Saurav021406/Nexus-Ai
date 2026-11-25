import pytest
from fastapi.testclient import TestClient

import app.routers.email_generator as email_router
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[email_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def test_email_draft_requires_ai_processing_consent(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("AI provider should not be called without consent")

    monkeypatch.setattr(email_router, "get_consensus_json", fail_if_called)
    response = TestClient(app).post(
        "/email-generator/generate",
        json={
            "purpose": "Request an update",
            "context": "The project is due Friday.",
            "consent_to_ai_processing": False,
        },
    )

    assert response.status_code == 400
    assert "permission" in response.json()["detail"]


def test_email_draft_returns_validated_subject_and_body(monkeypatch):
    def fake_consensus(prompt, **kwargs):
        assert "project is due Friday" in prompt
        assert "Alex Example" not in prompt
        assert "a warm tone" in prompt
        assert kwargs["tier"] == "fast"
        return {
            "subject": "Project status update",
            "body": "Hi Alex,\n\nCould you share a quick project update?",
            "_consensus": {"models_used": ["groq"]},
        }

    monkeypatch.setattr(email_router, "get_consensus_json", fake_consensus)
    response = TestClient(app).post(
        "/email-generator/generate",
        json={
            "purpose": "Request a project update",
            "context": "The project is due Friday.",
            "recipient": "Alex",
            "signature": "Best,\nAlex Example",
            "tone": "warm",
            "length": "concise",
            "consent_to_ai_processing": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "subject": "Project status update",
        "body": "Hi Alex,\n\nCould you share a quick project update?\n\nBest,\nAlex Example",
    }


def test_email_draft_rejects_context_over_limit():
    response = TestClient(app).post(
        "/email-generator/generate",
        json={
            "purpose": "Write an update",
            "context": "x" * 10_001,
            "consent_to_ai_processing": True,
        },
    )

    assert response.status_code == 422
