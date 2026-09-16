import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.routers.scenario as scenario_router
from app.main import app
from app.services import automl


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[scenario_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def _build_and_register_model():
    rng = np.random.default_rng(0)
    n = 300
    price = rng.uniform(10, 100, size=n)
    target = 5 * price + rng.normal(0, 5, size=n)
    df = pd.DataFrame({"price": price, "target": target})

    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, "target")
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, "target",
        user_id="user-1", dataset_id="d1", background_sample=X_test[:50],
    )
    return df.drop(columns=["target"]), model_id


def test_simulate_endpoint_returns_real_directional_result(monkeypatch):
    df, model_id = _build_and_register_model()
    monkeypatch.setattr(scenario_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        scenario_router.scenario_simulator,
        "explain_scenario",
        lambda result: {"summary": "s", "key_metrics": [], "recommendation": "r"},
    )

    client = TestClient(app)
    response = client.post(
        "/scenario/simulate",
        json={
            "dataset_id": "d1",
            "model_id": model_id,
            "adjustments": [{"column": "price", "type": "percent", "value": 50}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome_type"] == "numeric"
    assert body["scenario_mean"] > body["baseline_mean"]
    assert body["business_summary"]["summary"] == "s"


def test_simulate_endpoint_returns_400_for_unknown_column(monkeypatch):
    df, model_id = _build_and_register_model()
    monkeypatch.setattr(scenario_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post(
        "/scenario/simulate",
        json={
            "dataset_id": "d1",
            "model_id": model_id,
            "adjustments": [{"column": "does_not_exist", "type": "percent", "value": 10}],
        },
    )

    assert response.status_code == 400


def test_simulate_endpoint_returns_400_for_unknown_model(monkeypatch):
    df, _ = _build_and_register_model()
    monkeypatch.setattr(scenario_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post(
        "/scenario/simulate",
        json={
            "dataset_id": "d1",
            "model_id": "does-not-exist",
            "adjustments": [{"column": "price", "type": "percent", "value": 10}],
        },
    )

    assert response.status_code == 400
