import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.routers.root_cause as root_cause_router
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[root_cause_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def _build_revenue_drop_dataset() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    rows = []
    for i, date in enumerate(dates):
        is_after = i >= 30
        rows.append({"date": date, "region": "West", "revenue": 200 if is_after else 1000})
        rows.append({"date": date, "region": "East", "revenue": 980})
    return pd.DataFrame(rows)


def test_analyze_endpoint_finds_the_real_driver(monkeypatch):
    df = _build_revenue_drop_dataset()
    monkeypatch.setattr(root_cause_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        root_cause_router.root_cause,
        "explain_root_cause",
        lambda result: {"summary": "s", "key_metrics": ["m"], "recommendation": "r"},
    )

    client = TestClient(app)
    response = client.post(
        "/root-cause/analyze",
        json={"dataset_id": "d1", "metric_column": "revenue", "date_column": "date"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["overall_change"] < 0
    assert body["top_contributors"][0]["segment_value"] == "West"
    assert body["business_summary"]["summary"] == "s"


def test_analyze_endpoint_returns_400_for_unknown_metric_column(monkeypatch):
    df = _build_revenue_drop_dataset()
    monkeypatch.setattr(root_cause_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post(
        "/root-cause/analyze",
        json={"dataset_id": "d1", "metric_column": "does_not_exist", "date_column": "date"},
    )

    assert response.status_code == 400


def test_analyze_endpoint_respects_explicit_segment_columns(monkeypatch):
    df = _build_revenue_drop_dataset()
    df["extra_col"] = "always_the_same"
    monkeypatch.setattr(root_cause_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        root_cause_router.root_cause,
        "explain_root_cause",
        lambda result: {"summary": "s", "key_metrics": [], "recommendation": "r"},
    )

    client = TestClient(app)
    response = client.post(
        "/root-cause/analyze",
        json={
            "dataset_id": "d1",
            "metric_column": "revenue",
            "date_column": "date",
            "segment_columns": ["region"],
        },
    )

    assert response.status_code == 200
    assert response.json()["segment_columns_analyzed"] == ["region"]
