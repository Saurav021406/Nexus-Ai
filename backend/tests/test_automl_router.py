import numpy as np
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
    monkeypatch.setattr(
        automl_router.approvals, "create_version", lambda **kwargs: {"id": "v1", "version_number": 1}
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
    assert body["version_id"] == "v1"
    assert body["version_number"] == 1


def test_run_endpoint_with_explicit_regression_problem_type(monkeypatch):
    df = _regression_df()
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        automl_router.automl, "explain_results", lambda result: {"summary": "s", "key_metrics": [], "recommendation": "r"}
    )
    monkeypatch.setattr(
        automl_router.approvals, "create_version", lambda **kwargs: {"id": "v1", "version_number": 1}
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


def test_list_automl_versions_returns_history(monkeypatch):
    monkeypatch.setattr(
        automl_router.approvals,
        "list_versions",
        lambda resource_type, resource_id, user_id: [
            {
                "id": "v2",
                "version_number": 2,
                "created_at": "2026-01-02T00:00:00Z",
                "content": {
                    "problem_type": "classification",
                    "target_column": "churn",
                    "best_model_name": "XGBoost",
                    "models": [],
                },
            },
            {
                "id": "v1",
                "version_number": 1,
                "created_at": "2026-01-01T00:00:00Z",
                "content": {
                    "problem_type": "classification",
                    "target_column": "churn",
                    "best_model_name": "Random Forest",
                    "models": [],
                },
            },
        ],
    )

    client = TestClient(app)
    response = client.get("/automl/versions", params={"dataset_id": "d1"})

    assert response.status_code == 200
    versions = response.json()["versions"]
    assert len(versions) == 2
    assert versions[0]["version_number"] == 2
    assert versions[0]["best_model_name"] == "XGBoost"


# --- Batch CSV prediction --------------------------------------------------

def test_predict_csv_appends_prediction_column_and_returns_a_csv(monkeypatch):
    monkeypatch.setattr(
        automl_router.automl,
        "predict_with_model",
        lambda model_id, user_id, rows: {"predictions": [1, 0], "model_id": model_id, "n_rows": len(rows)},
    )

    csv_content = "feature_0,feature_1\n1.0,2.0\n3.0,4.0\n"
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "m1"},
        files={"file": ("new_data.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="new_data_predictions.csv"' in response.headers["content-disposition"]
    body = response.content.decode()
    assert "prediction" in body
    assert "1.0,2.0,1" in body
    assert "3.0,4.0,0" in body


def test_predict_csv_appends_probability_columns_for_classification(monkeypatch):
    monkeypatch.setattr(
        automl_router.automl,
        "predict_with_model",
        lambda model_id, user_id, rows: {
            "predictions": ["yes"],
            "model_id": model_id,
            "n_rows": 1,
            "probabilities": [{"yes": 0.8, "no": 0.2}],
        },
    )

    csv_content = "f0\n1.0\n"
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "m1"},
        files={"file": ("data.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.content.decode()
    assert "probability_yes" in body
    assert "probability_no" in body


def test_predict_csv_rejects_a_non_csv_file():
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "m1"},
        files={"file": ("data.txt", "not,a,csv", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_csv_rejects_an_empty_file():
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "m1"},
        files={"file": ("data.csv", "", "text/csv")},
    )
    assert response.status_code == 400


def test_predict_csv_rejects_a_csv_with_no_rows():
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "m1"},
        files={"file": ("data.csv", "col_a,col_b\n", "text/csv")},
    )
    assert response.status_code == 400


def test_predict_csv_rejects_when_the_underlying_model_lookup_fails(monkeypatch):
    def fail(model_id, user_id, rows):
        raise ValueError("model not found")

    monkeypatch.setattr(automl_router.automl, "predict_with_model", fail)

    csv_content = "f0\n1.0\n"
    client = TestClient(app)
    response = client.post(
        "/automl/predict/csv",
        data={"model_id": "does-not-exist"},
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 400


# --- Anomaly detection endpoint --------------------------------------------

def test_anomaly_endpoint_returns_real_anomalies(monkeypatch):
    rng = np.random.default_rng(0)
    normal_points = rng.normal(loc=0, scale=1, size=(190, 3))
    outliers = rng.normal(loc=50, scale=1, size=(10, 3))
    df = pd.DataFrame(np.vstack([normal_points, outliers]), columns=["a", "b", "c"])
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post("/automl/anomalies", json={"dataset_id": "d1"})

    assert response.status_code == 200
    body = response.json()
    assert body["n_anomalies"] > 0
    assert body["n_rows_analyzed"] == 200


def test_anomaly_endpoint_returns_400_for_insufficient_data(monkeypatch):
    df = pd.DataFrame({"only_non_numeric_col": ["a", "b", "c"] * 20})
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)

    client = TestClient(app)
    response = client.post("/automl/anomalies", json={"dataset_id": "d1"})

    assert response.status_code == 400
