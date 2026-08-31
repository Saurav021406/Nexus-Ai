import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification, make_regression

import app.routers.automl as automl_router
from app.main import app


class FakeUser:
    id = "user-1"


@pytest.fixture(autouse=True)
def _auth_override():
    app.dependency_overrides[automl_router.get_current_user] = lambda: FakeUser()
    yield
    app.dependency_overrides.clear()


def _classification_df():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=0)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y
    return df


def _regression_df():
    X, y = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=0)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y
    return df


def test_run_endpoint_trains_real_models_and_returns_full_shape(monkeypatch):
    df = _classification_df()
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        automl_router.automl, "explain_results", lambda result: {"summary": "s", "key_metrics": ["m"], "recommendation": "r"}
    )

    client = TestClient(app)
    response = client.post("/automl/run", json={"dataset_id": "d1", "target_column": "target"})

    assert response.status_code == 200
    body = response.json()
    assert body["problem_type"] == "classification"
    assert body["best_model_name"]
    assert len(body["models"]) >= 2
    assert body["business_summary"]["summary"] == "s"
    assert "shap_importances" in body


def test_run_endpoint_with_explicit_regression_problem_type(monkeypatch):
    df = _regression_df()
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        automl_router.automl, "explain_results", lambda result: {"summary": "s", "key_metrics": [], "recommendation": "r"}
    )

    client = TestClient(app)
    response = client.post(
        "/automl/run", json={"dataset_id": "d1", "target_column": "target", "problem_type": "regression"}
    )

    assert response.status_code == 200
    assert response.json()["problem_type"] == "regression"


def test_run_endpoint_returns_400_for_unknown_target_column(monkeypatch):
    df = _classification_df()
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post("/automl/run", json={"dataset_id": "d1", "target_column": "nope"})

    assert response.status_code == 400


def test_cluster_endpoint_returns_real_clusters(monkeypatch):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=0, cluster_std=0.5)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post("/automl/cluster", json={"dataset_id": "d1"})

    assert response.status_code == 200
    body = response.json()
    assert 2 <= body["n_clusters"] <= 8


def test_cluster_endpoint_returns_400_for_insufficient_data(monkeypatch):
    df = pd.DataFrame({"only_one_col": range(50)})
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post("/automl/cluster", json={"dataset_id": "d1"})

    assert response.status_code == 400
