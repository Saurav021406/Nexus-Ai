"""AutoML Engine (Phase 5 of the roadmap).

This replaces agents/ml_engineer.py's Phase-4-era placeholder - which
explicitly only ever REASONED about "is this dataset ready for modeling"
via an LLM, and never trained anything (see that file's own docstring:
"Do NOT implement the complete Phase 5 AutoML engine here"). This module
is that engine: genuine model training, cross-validation, and SHAP
explainability, using actual scikit-learn/XGBoost/LightGBM code - not an
LLM guessing at plausible-sounding numbers.

    dataset + target column
        -> detect_problem_type()      (classification vs regression - rule-based, free)
        -> train_and_compare()        (multiple real models, k-fold CV, held-out test metrics)
        -> compute_shap_importance()  (real SHAP values on the winning model)
        -> explain_results()          (Business Analyst LLM layer - plain English,
                                        grounded ONLY in the exact numbers above,
                                        same "never invent a number" discipline as
                                        every other specialist in this codebase)

Clustering and forecasting are deliberately NOT duplicated here:
forecasting already has its own real, working implementation (see
services powering /forecast - agents/ml_engineer.py's docstring already
points at this), and clustering has a much smaller, distinct shape
(no target column, no train/test split) that doesn't fit the
train-compare-explain pipeline below. See cluster_dataset() at the bottom
for a minimal, separate clustering path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.consensus import get_consensus_json

# XGBoost/LightGBM are optional at import time so a missing/broken native
# build degrades to "fewer candidate models" rather than crashing AutoML
# entirely - scikit-learn's own models are always enough to produce a result.
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# A numeric column with this few distinct values is almost always encoding
# categories (e.g. star ratings 1-5, a 0/1 flag) rather than a continuous
# quantity - classification fits it far better than regression.
MAX_NUMERIC_CLASSES_FOR_CLASSIFICATION = 15
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
RANDOM_STATE = 42
TOP_N_SHAP_FEATURES = 10


@dataclass
class ModelResult:
    name: str
    cv_score_mean: float
    cv_score_std: float
    cv_metric: str
    test_metrics: dict[str, float]


@dataclass
class AutoMLResult:
    problem_type: str  # "classification" | "regression"
    target_column: str
    feature_columns: list[str]
    n_rows_used: int
    n_rows_dropped: int  # rows dropped for missing target
    models: list[ModelResult]
    best_model_name: str
    primary_metric: str
    shap_importances: list[dict[str, Any]] = field(default_factory=list)
    shap_unavailable_reason: str | None = None


def detect_problem_type(target: pd.Series) -> str:
    """Rule-based, free, instant - no LLM call, same "no token cost for
    something code can determine exactly" philosophy as Domain Gate and
    Evidence Gate elsewhere in this codebase.

    - Non-numeric (object/category/bool) target -> classification, always.
    - Numeric target with few distinct values -> classification (it's
      almost certainly encoding categories, e.g. a 1-5 rating or a 0/1 flag).
    - Numeric target with many distinct values -> regression.
    """
    non_null = target.dropna()
    if non_null.empty:
        raise ValueError("Target column has no non-null values to learn from.")

    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"

    if non_null.nunique() <= MAX_NUMERIC_CLASSES_FOR_CLASSIFICATION:
        return "classification"

    return "regression"


def _build_feature_matrix(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, pd.Series, list[str]]:
    """Drops rows with a missing target (can't learn from an unlabeled
    example), imputes missing feature values (median for numeric, most
    frequent for categorical), one-hot encodes categorical features, and
    scales numeric features - a plain, transparent pipeline rather than a
    black-box sklearn Pipeline object, so the feature names line up
    exactly with what SHAP reports back later."""
    working = df.dropna(subset=[target_column]).copy()
    y = working[target_column]
    X_raw = working.drop(columns=[target_column])

    numeric_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    pieces = []
    feature_names: list[str] = []

    if numeric_cols:
        numeric_imputed = SimpleImputer(strategy="median").fit_transform(X_raw[numeric_cols])
        numeric_scaled = StandardScaler().fit_transform(numeric_imputed)
        pieces.append(numeric_scaled)
        feature_names.extend(numeric_cols)

    if categorical_cols:
        cat_imputed = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[categorical_cols].astype(str))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
        cat_encoded = encoder.fit_transform(cat_imputed)
        pieces.append(cat_encoded)
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())

    if not pieces:
        raise ValueError("No usable feature columns remain after removing the target column.")

    X = np.hstack(pieces)
    return X, y, feature_names


def _candidate_models(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        }
        if _HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
            )
        if _HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)
        return models

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0)
    if _HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)
    return models


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    average = "binary" if len(set(y_true)) == 2 else "macro"
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def train_and_compare(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str | None = None,
    cv_folds: int = DEFAULT_CV_FOLDS,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[AutoMLResult, Any, np.ndarray, list[str]]:
    """Trains every candidate model for the detected/given problem type,
    cross-validates each on the training split, evaluates all of them on
    a held-out test split, and picks the best by the primary metric
    (accuracy for classification, R^2 for regression).

    Returns (result, best_fitted_model, X_test, feature_names) - the
    fitted model and X_test are returned alongside the serializable
    result so compute_shap_importance() can use them without retraining.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in this dataset.")

    n_before = len(df)
    resolved_problem_type = problem_type or detect_problem_type(df[target_column])
    if resolved_problem_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported problem_type '{resolved_problem_type}' - use 'classification' or 'regression'.")

    X, y, feature_names = _build_feature_matrix(df, target_column)
    n_after = len(y)

    if n_after < 20:
        raise ValueError(
            f"Only {n_after} usable rows after dropping missing targets - too few to train and "
            "evaluate a model reliably (need at least 20)."
        )

    stratify = y if resolved_problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    candidates = _candidate_models(resolved_problem_type)
    cv_metric = "accuracy" if resolved_problem_type == "classification" else "r2"
    splitter = (
        StratifiedKFold(n_splits=min(cv_folds, y_train.value_counts().min()), shuffle=True, random_state=RANDOM_STATE)
        if resolved_problem_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    )

    model_results: list[ModelResult] = []
    fitted_models: dict[str, Any] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn/xgboost convergence & deprecation noise, not actionable here
        for name, model in candidates.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=splitter, scoring=cv_metric)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_metrics = (
                    _classification_metrics(y_test, y_pred)
                    if resolved_problem_type == "classification"
                    else _regression_metrics(y_test, y_pred)
                )
                model_results.append(
                    ModelResult(
                        name=name,
                        cv_score_mean=round(float(cv_scores.mean()), 4),
                        cv_score_std=round(float(cv_scores.std()), 4),
                        cv_metric=cv_metric,
                        test_metrics=test_metrics,
                    )
                )
                fitted_models[name] = model
            except Exception as e:
                # One model failing (e.g. a solver convergence edge case)
                # shouldn't take down the whole comparison - just exclude
                # it and keep going with whatever did train successfully.
                model_results.append(
                    ModelResult(name=name, cv_score_mean=float("nan"), cv_score_std=float("nan"), cv_metric=cv_metric, test_metrics={"error": str(e)})
                )

    successful = [m for m in model_results if not np.isnan(m.cv_score_mean)]
    if not successful:
        raise RuntimeError("Every candidate model failed to train on this dataset.")

    best = max(successful, key=lambda m: m.cv_score_mean)

    result = AutoMLResult(
        problem_type=resolved_problem_type,
        target_column=target_column,
        feature_columns=feature_names,
        n_rows_used=n_after,
        n_rows_dropped=n_before - n_after,
        models=model_results,
        best_model_name=best.name,
        primary_metric=cv_metric,
    )

    return result, fitted_models[best.name], X_test, feature_names


def compute_shap_importance(model: Any, X_test: np.ndarray, feature_names: list[str], max_features: int = TOP_N_SHAP_FEATURES) -> tuple[list[dict[str, Any]], str | None]:
    """Real SHAP values on the actual winning model - not a proxy like
    sklearn's built-in .feature_importances_, which only exists for
    tree models and doesn't account for feature interactions the way
    SHAP does. Falls back gracefully (empty list + a reason string,
    never an exception) if SHAP isn't installed or the model type isn't
    one SHAP's fast explainers support - explainability is a bonus on
    top of a working result, not a hard requirement for one."""
    if not _HAS_SHAP:
        return [], "SHAP is not installed in this environment."

    try:
        sample = X_test if len(X_test) <= 200 else X_test[np.random.default_rng(RANDOM_STATE).choice(len(X_test), 200, replace=False)]

        model_type = type(model).__name__
        if any(name in model_type for name in ("RandomForest", "XGB", "LGBM")):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.LinearExplainer(model, sample)
            shap_values = explainer.shap_values(sample)

        # SHAP's return shape varies by version and problem type:
        # - list of arrays (one per class, older SHAP / some multiclass cases)
        # - 3D ndarray (n_samples, n_features, n_classes) - newer SHAP's
        #   TreeExplainer output for classifiers
        # - 2D ndarray (n_samples, n_features) - regression, or binary
        #   classification already reduced to one class
        # All three are normalized down to one (n_features,) importance
        # vector by averaging absolute magnitude across classes, then
        # across samples.
        if isinstance(shap_values, list):
            abs_values = np.mean([np.abs(v) for v in shap_values], axis=0)
        elif shap_values.ndim == 3:
            abs_values = np.abs(shap_values).mean(axis=2)
        else:
            abs_values = np.abs(shap_values)

        mean_abs_importance = abs_values.mean(axis=0)
        ranked = sorted(zip(feature_names, mean_abs_importance), key=lambda pair: pair[1], reverse=True)

        return (
            [{"feature": name, "importance": round(float(value), 6)} for name, value in ranked[:max_features]],
            None,
        )
    except Exception as e:
        return [], f"SHAP computation failed for this model type: {e}"


def explain_results(result: AutoMLResult) -> dict:
    """The 'Business Analyst layer' the roadmap explicitly calls for:
    plain-English insights, not '93% accuracy' - grounded ONLY in the
    exact numbers already computed above (same never-invent-a-number
    discipline every other specialist in this codebase follows), never
    re-estimating or guessing at a metric."""
    best = next(m for m in result.models if m.name == result.best_model_name)
    models_summary = "\n".join(
        f"- {m.name}: cross-validated {m.cv_metric} = {m.cv_score_mean} (+/- {m.cv_score_std}), "
        f"test set metrics = {m.test_metrics}"
        for m in result.models
        if not np.isnan(m.cv_score_mean)
    )
    shap_summary = (
        "Top features by SHAP importance: "
        + ", ".join(f"{f['feature']} ({f['importance']})" for f in result.shap_importances)
        if result.shap_importances
        else "SHAP feature importance was not available for this model."
    )

    prompt = f"""You are a Business Analyst translating machine learning results for a
non-technical stakeholder. Use ONLY the exact numbers given below - never invent, round
differently, or estimate a number that isn't present here.

Problem type: {result.problem_type}
Target column: {result.target_column}
Rows used for training/evaluation: {result.n_rows_used} (dropped {result.n_rows_dropped} rows with a missing target)
Best model: {result.best_model_name}

MODEL COMPARISON:
{models_summary}

{shap_summary}

Write plain-English business insight, NOT a technical accuracy readout. For example,
instead of "93% accuracy", say something like "the model correctly identifies about 9
out of every 10 cases" - translate metrics into concrete, real-world terms tied to what
the target column actually represents. Explain what the top features mean for decision-
making, not just that they're "important".

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph translating the best model's performance into plain business terms using only exact numbers above",
  "key_metrics": ["plain-English restatement of a metric 1", "metric 2", "metric 3"],
  "recommendation": "one concrete next step (e.g. collect more of X, investigate Y feature further, deploy for Z use case)"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)


def cluster_dataset(df: pd.DataFrame, feature_columns: list[str] | None = None, max_k: int = 8) -> dict:
    """Minimal clustering path (no target column, no train/test split -
    a genuinely different shape than train_and_compare above). Picks k
    automatically via silhouette score across k=2..max_k, so the user
    doesn't have to guess a cluster count."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    numeric_df = df[feature_columns] if feature_columns else df.select_dtypes(include="number")
    numeric_df = numeric_df.dropna()
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns with no missing values to cluster on.")
    if len(numeric_df) < 20:
        raise ValueError(f"Only {len(numeric_df)} usable rows - too few to cluster reliably (need at least 20).")

    X = StandardScaler().fit_transform(numeric_df)

    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, min(max_k, len(numeric_df) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    cluster_sizes = pd.Series(best_labels).value_counts().sort_index()
    cluster_profiles = numeric_df.assign(_cluster=best_labels).groupby("_cluster").mean().round(3)

    return {
        "n_clusters": best_k,
        "silhouette_score": round(float(best_score), 4),
        "cluster_sizes": cluster_sizes.to_dict(),
        "cluster_profiles": cluster_profiles.to_dict(orient="index"),
        "feature_columns": list(numeric_df.columns),
    }
