import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression

from app.services import automl


def _classification_df(n_samples=200, n_features=5, n_classes=2, random_state=0) -> tuple[pd.DataFrame, str]:
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=3, n_classes=n_classes, random_state=random_state
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, "target"


def _regression_df(n_samples=200, n_features=5, random_state=0) -> tuple[pd.DataFrame, str]:
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=5.0, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, "target"


# --- detect_problem_type ---------------------------------------------

def test_detects_classification_for_binary_numeric_target():
    target = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    assert automl.detect_problem_type(target) == "classification"


def test_detects_classification_for_string_target():
    target = pd.Series(["cat", "dog", "cat", "bird"] * 5)
    assert automl.detect_problem_type(target) == "classification"


def test_detects_regression_for_continuous_numeric_target():
    target = pd.Series(np.random.default_rng(0).normal(100, 20, size=50))
    assert automl.detect_problem_type(target) == "regression"


def test_raises_on_all_null_target():
    target = pd.Series([None, None, None])
    with pytest.raises(ValueError):
        automl.detect_problem_type(target)


# --- train_and_compare: classification --------------------------------

def test_classification_trains_multiple_models_and_picks_a_best_one():
    df, target_col = _classification_df()
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)

    assert result.problem_type == "classification"
    assert len(result.models) >= 2  # at minimum LogisticRegression + RandomForest
    assert result.best_model_name in {m.name for m in result.models}
    assert result.primary_metric == "accuracy"
    best = next(m for m in result.models if m.name == result.best_model_name)
    assert 0.0 <= best.cv_score_mean <= 1.0
    assert "accuracy" in best.test_metrics
    assert fitted_model is not None
    assert len(feature_names) == 5


def test_classification_test_metrics_are_genuinely_computed_not_placeholders():
    # A clean, easily-separable synthetic classification problem should
    # score well above chance (0.5) - this is a real check that models
    # are actually being trained and evaluated, not stubbed.
    df, target_col = _classification_df(n_samples=300, n_features=6)
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    best = next(m for m in result.models if m.name == result.best_model_name)
    assert best.test_metrics["accuracy"] > 0.6


# --- train_and_compare: regression -------------------------------------

def test_regression_trains_multiple_models_and_picks_a_best_one():
    df, target_col = _regression_df()
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)

    assert result.problem_type == "regression"
    assert len(result.models) >= 2
    assert result.primary_metric == "r2"
    best = next(m for m in result.models if m.name == result.best_model_name)
    assert "r2" in best.test_metrics
    assert "rmse" in best.test_metrics


def test_regression_r2_is_meaningfully_positive_on_a_learnable_signal():
    df, target_col = _regression_df(n_samples=300)
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    best = next(m for m in result.models if m.name == result.best_model_name)
    assert best.test_metrics["r2"] > 0.5


# --- categorical features & missing data --------------------------------

def test_handles_mixed_numeric_and_categorical_features():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "numeric_a": rng.normal(size=100),
        "numeric_b": rng.normal(size=100),
        "category": rng.choice(["A", "B", "C"], size=100),
        "target": rng.integers(0, 2, size=100),
    })
    result, _, _, feature_names, _ = automl.train_and_compare(df, "target")
    assert result.n_rows_used == 100
    # one-hot encoding expands "category" into multiple feature columns
    assert len(feature_names) > 3


def test_drops_rows_with_missing_target_and_reports_the_count():
    df, target_col = _classification_df(n_samples=100)
    df.loc[:9, target_col] = np.nan  # 10 rows with missing target
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    assert result.n_rows_dropped == 10
    assert result.n_rows_used == 90


def test_raises_on_unknown_target_column():
    df, _ = _classification_df()
    with pytest.raises(ValueError):
        automl.train_and_compare(df, "does_not_exist")


def test_raises_on_too_few_usable_rows():
    df, target_col = _classification_df(n_samples=10)
    with pytest.raises(ValueError):
        automl.train_and_compare(df, target_col)


# --- SHAP explainability -------------------------------------------------

def test_shap_importance_returns_ranked_real_features():
    df, target_col = _classification_df(n_samples=300, n_features=6)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)

    importances, reason = automl.compute_shap_importance(fitted_model, X_test, feature_names)

    assert reason is None
    assert len(importances) > 0
    assert all(f["feature"] in feature_names for f in importances)
    # ranked descending by importance
    values = [f["importance"] for f in importances]
    assert values == sorted(values, reverse=True)


def test_shap_importance_on_regression_model():
    df, target_col = _regression_df(n_samples=300)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)
    importances, reason = automl.compute_shap_importance(fitted_model, X_test, feature_names)
    assert reason is None
    assert len(importances) > 0


# --- explain_results (Business Analyst layer, LLM call mocked) ----------

def test_explain_results_calls_llm_with_grounded_numbers(monkeypatch):
    df, target_col = _classification_df(n_samples=200)
    result, _, _, _, _ = automl.train_and_compare(df, target_col)

    captured_prompt = {}

    def fake_consensus_json(prompt, **kwargs):
        captured_prompt["prompt"] = prompt
        return {"summary": "fake", "key_metrics": ["fake"], "recommendation": "fake"}

    monkeypatch.setattr(automl, "get_consensus_json", fake_consensus_json)

    output = automl.explain_results(result)

    assert output["summary"] == "fake"
    # the exact best model name and target column must appear in the
    # prompt - grounding check, same discipline as every other specialist
    assert result.best_model_name in captured_prompt["prompt"]
    assert result.target_column in captured_prompt["prompt"]
    assert "93%" in captured_prompt["prompt"]  # the explicit anti-pattern instruction


# --- clustering -----------------------------------------------------------

def test_cluster_dataset_finds_a_reasonable_k():
    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=0, cluster_std=0.5)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    result = automl.cluster_dataset(df)

    assert 2 <= result["n_clusters"] <= 8
    assert result["silhouette_score"] > 0.3  # well-separated synthetic blobs should score decently
    assert sum(result["cluster_sizes"].values()) == 150


def test_cluster_dataset_raises_with_too_few_numeric_columns():
    df = pd.DataFrame({"only_one_numeric_col": range(50)})
    with pytest.raises(ValueError):
        automl.cluster_dataset(df)


def test_cluster_dataset_raises_with_too_few_rows():
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    with pytest.raises(ValueError):
        automl.cluster_dataset(df)
