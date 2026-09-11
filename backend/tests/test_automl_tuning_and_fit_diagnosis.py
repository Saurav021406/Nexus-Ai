import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_regression

from app.services import automl


def _classification_df(n_samples=200, n_features=6, random_state=0):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=4, n_redundant=0, random_state=random_state
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, "target"


def _regression_df(n_samples=200, n_features=5, random_state=0):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=5.0, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, "target"


# --- Hyperparameter tuning -------------------------------------------------

def test_tuning_disabled_by_default_leaves_every_model_untuned():
    df, target_col = _classification_df()
    result, _, _, _, _ = automl.train_and_compare(df, target_col)  # tune_hyperparameters not passed - defaults False
    assert all(m.tuned_params is None for m in result.models)


def test_tuning_enabled_populates_tuned_params_for_eligible_models():
    df, target_col = _classification_df(n_samples=200, n_features=6)
    result, _, _, _, _ = automl.train_and_compare(df, target_col, tune_hyperparameters=True)

    tunable_names = {"Logistic Regression", "Random Forest", "XGBoost", "LightGBM"}
    for m in result.models:
        if m.name in tunable_names:
            assert m.tuned_params is not None and len(m.tuned_params) > 0, f"{m.name} should have tuned params"


def test_tuning_never_assigns_params_to_linear_regression():
    # Linear Regression has nothing meaningful to search - it should
    # never even get an Optuna study, let alone tuned_params.
    df, target_col = _regression_df(n_samples=150, n_features=5)
    result, _, _, _, _ = automl.train_and_compare(df, target_col, tune_hyperparameters=True)

    linear = next(m for m in result.models if m.name == "Linear Regression")
    assert linear.tuned_params is None


def test_tuned_params_actually_used_are_within_the_declared_search_space():
    df, target_col = _classification_df(n_samples=200, n_features=6)
    result, _, _, _, _ = automl.train_and_compare(df, target_col, tune_hyperparameters=True)

    rf = next(m for m in result.models if m.name == "Random Forest")
    assert 50 <= rf.tuned_params["n_estimators"] <= 300
    assert 3 <= rf.tuned_params["max_depth"] <= 20


def test_response_reports_whether_tuning_ran(monkeypatch):
    # Router-level: tuning_enabled should reflect exactly what was requested.
    import app.routers.automl as automl_router
    from fastapi.testclient import TestClient
    from app.main import app

    df, target_col = _classification_df(n_samples=150, n_features=4)
    app.dependency_overrides[automl_router.get_current_user] = lambda: type("U", (), {"id": "u1"})()
    monkeypatch.setattr(automl_router, "get_dataset_dataframe", lambda dataset_id, user_id: df)
    monkeypatch.setattr(
        automl_router.automl, "explain_results", lambda result: {"summary": "s", "key_metrics": [], "recommendation": "r"}
    )
    monkeypatch.setattr(automl_router.approvals, "create_version", lambda **kwargs: {"id": "v1", "version_number": 1})

    try:
        client = TestClient(app)
        response = client.post(
            "/automl/run",
            json={"dataset_id": "d1", "target_column": target_col, "tune_hyperparameters": False},
        )
        assert response.status_code == 200
        assert response.json()["tuning_enabled"] is False
    finally:
        app.dependency_overrides.clear()


# --- Overfitting / underfitting detection -----------------------------------

def test_logistic_regression_underfits_a_genuinely_nonlinear_target():
    # Concentric circles are impossible for a linear decision boundary to
    # separate well - Logistic Regression should score barely above
    # chance on BOTH train and test, which is the underfitting signature
    # (as opposed to overfitting, where train would be high and test low).
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)
    df = pd.DataFrame(X, columns=["f0", "f1"])
    df["target"] = y

    result, _, _, _, _ = automl.train_and_compare(df, "target")

    logistic = next(m for m in result.models if m.name == "Logistic Regression")
    assert logistic.fit_diagnosis == "underfitting"
    assert logistic.train_score < 0.6  # genuinely can't even fit the training data well

    # a non-linear model on the SAME data should NOT underfit - confirms
    # the diagnosis is about the model/data fit, not a bug that flags
    # everything as underfitting regardless of actual performance
    tree_model = next(m for m in result.models if m.name == "Random Forest")
    assert tree_model.fit_diagnosis == "good_fit"


def test_tree_model_overfits_a_tiny_high_dimensional_pure_noise_target():
    # A target with NO real signal, tiny sample, many features - a
    # flexible tree model will memorize the training data (train_score
    # near 1.0) while generalizing no better than chance on the test set.
    # This is the textbook overfitting signature: high train, low test.
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"f{i}" for i in range(20)])
    df["target"] = rng.integers(0, 2, size=n)

    result, _, _, _, _ = automl.train_and_compare(df, "target")

    rf = next(m for m in result.models if m.name == "Random Forest")
    assert rf.fit_diagnosis == "overfitting"
    assert rf.train_score > 0.9
    assert rf.test_metrics["accuracy"] < rf.train_score - automl.OVERFIT_GAP_THRESHOLD


def test_overfitting_best_model_triggers_a_result_level_warning():
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"f{i}" for i in range(20)])
    df["target"] = rng.integers(0, 2, size=n)

    result, _, _, _, _ = automl.train_and_compare(df, "target")
    best = next(m for m in result.models if m.name == result.best_model_name)

    if best.fit_diagnosis == "overfitting":
        assert any("overfit" in w.lower() for w in result.warnings)


def test_well_fitted_model_has_no_fit_diagnosis_warning():
    df, target_col = _classification_df(n_samples=300, n_features=6)
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    best = next(m for m in result.models if m.name == result.best_model_name)

    if best.fit_diagnosis == "good_fit":
        assert not any("overfit" in w.lower() or "underfit" in w.lower() for w in result.warnings)


def test_diagnose_fit_function_directly_for_all_three_outcomes():
    assert automl._diagnose_fit(train_score=0.4, test_score=0.35, problem_type="classification") == "underfitting"
    assert automl._diagnose_fit(train_score=0.99, test_score=0.6, problem_type="classification") == "overfitting"
    assert automl._diagnose_fit(train_score=0.85, test_score=0.8, problem_type="classification") == "good_fit"
    # regression uses a different (lower) underfit threshold
    assert automl._diagnose_fit(train_score=0.2, test_score=0.15, problem_type="regression") == "underfitting"
    assert automl._diagnose_fit(train_score=0.9, test_score=0.5, problem_type="regression") == "overfitting"
