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
        -> _build_feature_matrix()    (impute/scale/encode + free ID-column and
                                        leakage/imbalance diagnostics, no LLM call)
        -> train_and_compare()        (multiple real models, k-fold CV, held-out test metrics)
        -> compute_shap_importance()  (real SHAP values on the winning model)
        -> explain_results()          (Business Analyst LLM layer - plain English,
                                        grounded ONLY in the exact numbers above,
                                        same "never invent a number" discipline as
                                        every other specialist in this codebase)
        -> register_model() / predict_with_model()
                                       (persist the winning model + its exact
                                        preprocessing so it can actually be
                                        used on new rows later, not just
                                        reported on once and discarded)

Clustering and forecasting are deliberately NOT duplicated here:
forecasting already has its own real, working implementation (see
services powering /forecast - agents/ml_engineer.py's docstring already
points at this), and clustering has a much smaller, distinct shape
(no target column, no train/test split) that doesn't fit the
train-compare-explain pipeline below. See cluster_dataset() at the bottom
for a minimal, separate clustering path.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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

# A column where every value is unique is almost certainly an identifier
# (customer_id, transaction UUID, row number) rather than a real signal -
# including it as a feature either adds pure noise (a fresh dummy per row)
# or, worse, lets a model "memorize" via a leaked ordering. Only auto-
# excluded when non-numeric OR the name itself looks like an id, so a
# genuinely continuous numeric measurement that happens to be all-unique
# (e.g. a precise sensor reading) is not mistakenly dropped.
ID_LIKE_NAME_PATTERN = re.compile(r"(^id$|_id$|^uuid$|_uuid$|^guid$)", re.IGNORECASE)

# A feature this correlated with the target is far more likely to be a
# leak (derived from or a proxy for the target itself) than a genuinely
# strong, legitimate predictor - flagged as a warning, not auto-excluded,
# since the user's judgment about their own data matters here.
LEAKAGE_CORRELATION_THRESHOLD = 0.95

# Ratio of majority-to-minority class count above which a classification
# target is considered imbalanced enough to warrant class_weight="balanced"
# and a warning - below this, plain accuracy is a fair metric on its own.
CLASS_IMBALANCE_RATIO_THRESHOLD = 3.0

# Trained models are kept in memory only (same single-process trade-off
# already made for services/cache.py - see that file's docstring) so a
# server restart clears them and retraining is needed. TTL keeps memory
# bounded if predict() is never called to clean things up itself.
MODEL_REGISTRY_TTL_SECONDS = 3600


@dataclass
class ModelResult:
    name: str
    cv_score_mean: float
    cv_score_std: float
    cv_metric: str
    test_metrics: dict[str, float]


@dataclass
class FeatureTransformer:
    """Everything needed to turn a NEW raw row into the exact same feature
    space the model was trained on - fitted once during training, then
    reused unchanged at prediction time. Keeping this as a plain object
    (rather than a black-box sklearn Pipeline) means the fitted pieces are
    individually inspectable and the feature name order is guaranteed to
    match what SHAP reports."""

    numeric_cols: list[str]
    categorical_cols: list[str]
    excluded_columns: list[str]
    numeric_imputer: Any | None
    scaler: Any | None
    categorical_imputer: Any | None
    encoder: Any | None
    target_encoder: Any | None  # LabelEncoder, only set for a non-numeric classification target

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        working = df.drop(columns=[c for c in self.excluded_columns if c in df.columns], errors="ignore")
        pieces = []

        if self.numeric_cols:
            numeric_part = working.reindex(columns=self.numeric_cols)
            numeric_imputed = self.numeric_imputer.transform(numeric_part)
            pieces.append(self.scaler.transform(numeric_imputed))

        if self.categorical_cols:
            cat_part = working.reindex(columns=self.categorical_cols).astype(str)
            cat_imputed = self.categorical_imputer.transform(cat_part)
            pieces.append(self.encoder.transform(cat_imputed))

        if not pieces:
            raise ValueError("No usable feature columns found in the input to predict on.")

        return np.hstack(pieces)

    def decode_target(self, encoded_predictions: np.ndarray) -> list:
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(encoded_predictions).tolist()
        return list(encoded_predictions)


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
    warnings: list[str] = field(default_factory=list)
    excluded_id_columns: list[str] = field(default_factory=list)
    class_imbalance: dict[str, Any] | None = None
    model_id: str | None = None


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


def _detect_id_like_columns(X_raw: pd.DataFrame) -> list[str]:
    """Free, instant, no LLM call - same philosophy as Domain Gate/Evidence
    Gate elsewhere in this codebase. A column is treated as an identifier
    (and excluded from features) only when it's unique per row AND either
    non-numeric or its name looks like an id - a purely-numeric, all-
    unique column with an ordinary name (e.g. a precise price or sensor
    reading) is left alone, since uniqueness alone isn't enough evidence."""
    n = len(X_raw)
    if n == 0:
        return []

    excluded = []
    for col in X_raw.columns:
        if X_raw[col].nunique(dropna=True) != n:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(X_raw[col])
        looks_like_id = bool(ID_LIKE_NAME_PATTERN.search(str(col)))
        if not is_numeric or looks_like_id:
            excluded.append(col)
    return excluded


def _detect_leakage_warnings(numeric_cols: list[str], X_raw: pd.DataFrame, y_numeric: pd.Series) -> list[str]:
    """Free, instant correlation check between each numeric feature and
    the (numeric-encoded) target - flagged as a warning rather than
    auto-excluded, since a very strong legitimate predictor looks
    identical to a leak from a pure-correlation standpoint and the
    user's own judgment about their data matters here."""
    warnings_found = []
    for col in numeric_cols:
        try:
            corr = X_raw[col].corr(y_numeric)
        except Exception:
            continue
        if corr is not None and abs(corr) >= LEAKAGE_CORRELATION_THRESHOLD:
            warnings_found.append(
                f"'{col}' is {abs(corr):.2f} correlated with the target - this may be a data leak "
                "(a feature derived from or that duplicates the target) rather than a genuine predictor. "
                "Worth double-checking before trusting this model."
            )
    return warnings_found


def _build_feature_matrix(
    df: pd.DataFrame, target_column: str, problem_type: str
) -> tuple[np.ndarray, pd.Series, list[str], FeatureTransformer, list[str], list[str], dict[str, Any] | None]:
    """Drops rows with a missing target (can't learn from an unlabeled
    example), excludes ID-like columns, imputes missing feature values
    (median for numeric, most frequent for categorical), one-hot encodes
    categorical features, and scales numeric features - a plain,
    transparent pipeline rather than a black-box sklearn Pipeline object,
    so the feature names line up exactly with what SHAP reports later and
    the fitted pieces can be reused as-is for prediction on new rows.

    Returns (X, y, feature_names, transformer, excluded_id_columns,
    leakage_warnings, class_imbalance)."""
    working = df.dropna(subset=[target_column]).copy()
    y = working[target_column]
    X_raw = working.drop(columns=[target_column])

    excluded_id_columns = _detect_id_like_columns(X_raw)
    X_raw = X_raw.drop(columns=excluded_id_columns)

    numeric_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    # Leakage check needs a numeric target - label-encode a non-numeric
    # classification target purely for this correlation check (not used
    # for training itself).
    y_numeric = y if pd.api.types.is_numeric_dtype(y) else pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    leakage_warnings = _detect_leakage_warnings(numeric_cols, X_raw, y_numeric)

    class_imbalance = None
    target_encoder = None
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y), index=y.index)
        counts = y.value_counts()
        if len(counts) >= 2:
            ratio = counts.max() / counts.min()
            if ratio >= CLASS_IMBALANCE_RATIO_THRESHOLD:
                class_imbalance = {
                    "detected": True,
                    "ratio": round(float(ratio), 2),
                    "class_counts": {str(k): int(v) for k, v in counts.items()},
                }

    pieces = []
    feature_names: list[str] = []
    numeric_imputer = scaler = categorical_imputer = encoder = None

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="median")
        numeric_imputed = numeric_imputer.fit_transform(X_raw[numeric_cols])
        scaler = StandardScaler()
        pieces.append(scaler.fit_transform(numeric_imputed))
        feature_names.extend(numeric_cols)

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputed = categorical_imputer.fit_transform(X_raw[categorical_cols].astype(str))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
        pieces.append(encoder.fit_transform(cat_imputed))
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())

    if not pieces:
        raise ValueError("No usable feature columns remain after removing the target and any ID-like columns.")

    X = np.hstack(pieces)
    transformer = FeatureTransformer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        excluded_columns=excluded_id_columns + [target_column],
        numeric_imputer=numeric_imputer,
        scaler=scaler,
        categorical_imputer=categorical_imputer,
        encoder=encoder,
        target_encoder=target_encoder,
    )
    return X, y, feature_names, transformer, excluded_id_columns, leakage_warnings, class_imbalance


def _candidate_models(problem_type: str, y_train: pd.Series | None = None, imbalanced: bool = False) -> dict[str, Any]:
    if problem_type == "classification":
        class_weight = "balanced" if imbalanced else None
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight=class_weight),
        }
        if _HAS_XGBOOST:
            # XGBoost has no class_weight param - scale_pos_weight is its
            # equivalent, but only applies to strictly BINARY classification.
            # Multiclass imbalance is still flagged in the result's warnings,
            # just not auto-corrected for this particular model.
            xgb_kwargs: dict[str, Any] = dict(n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
            if imbalanced and y_train is not None and y_train.nunique() == 2:
                counts = y_train.value_counts()
                xgb_kwargs["scale_pos_weight"] = float(counts.max() / counts.min())
            models["XGBoost"] = XGBClassifier(**xgb_kwargs)
        if _HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1, class_weight=class_weight)
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
) -> tuple[AutoMLResult, Any, np.ndarray, list[str], FeatureTransformer]:
    """Trains every candidate model for the detected/given problem type,
    cross-validates each on the training split, evaluates all of them on
    a held-out test split, and picks the best by the primary metric
    (accuracy for classification, R^2 for regression).

    Returns (result, best_fitted_model, X_test, feature_names,
    transformer) - the fitted model and X_test let compute_shap_importance()
    explain the winner without retraining, and the transformer lets
    register_model()/predict_with_model() apply the exact same
    preprocessing to brand new rows later.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in this dataset.")

    n_before = len(df)
    resolved_problem_type = problem_type or detect_problem_type(df[target_column])
    if resolved_problem_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported problem_type '{resolved_problem_type}' - use 'classification' or 'regression'.")

    X, y, feature_names, transformer, excluded_id_columns, leakage_warnings, class_imbalance = _build_feature_matrix(
        df, target_column, resolved_problem_type
    )
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

    is_imbalanced = class_imbalance is not None
    candidates = _candidate_models(resolved_problem_type, y_train=y_train, imbalanced=is_imbalanced)
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

    result_warnings = list(leakage_warnings)
    if excluded_id_columns:
        result_warnings.append(
            f"Excluded {len(excluded_id_columns)} likely ID column(s) from training: {', '.join(excluded_id_columns)}."
        )
    if is_imbalanced:
        result_warnings.append(
            f"The target is imbalanced (largest class is {class_imbalance['ratio']}x the smallest) - "
            "models were trained with class_weight=\"balanced\" to compensate, but accuracy alone can "
            "still be misleading here; check precision/recall per class."
        )

    result = AutoMLResult(
        problem_type=resolved_problem_type,
        target_column=target_column,
        feature_columns=feature_names,
        n_rows_used=n_after,
        n_rows_dropped=n_before - n_after,
        models=model_results,
        best_model_name=best.name,
        primary_metric=cv_metric,
        warnings=result_warnings,
        excluded_id_columns=excluded_id_columns,
        class_imbalance=class_imbalance,
    )

    return result, fitted_models[best.name], X_test, feature_names, transformer


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


_model_registry_lock = threading.Lock()
# model_id -> {model, transformer, feature_names, problem_type, target_column,
#              user_id, dataset_id, expires_at}
_model_registry: dict[str, dict[str, Any]] = {}


def _evict_expired_models_locked() -> None:
    """Caller must already hold _model_registry_lock. Same TTL-eviction
    pattern as services/cache.py, so a predict() that's never called still
    doesn't leak memory forever."""
    now = time.time()
    expired = [model_id for model_id, entry in _model_registry.items() if entry["expires_at"] < now]
    for model_id in expired:
        del _model_registry[model_id]


def register_model(
    model: Any,
    transformer: FeatureTransformer,
    feature_names: list[str],
    problem_type: str,
    target_column: str,
    user_id: str,
    dataset_id: str,
    ttl_seconds: int = MODEL_REGISTRY_TTL_SECONDS,
) -> str:
    """Persists a freshly-trained model + its exact preprocessing so
    predict_with_model() can use it later on new rows, without retraining.
    In-memory only (see MODEL_REGISTRY_TTL_SECONDS above for the trade-off
    this accepts) - a server restart clears it, same as services/cache.py."""
    model_id = str(uuid.uuid4())
    with _model_registry_lock:
        _evict_expired_models_locked()
        _model_registry[model_id] = {
            "model": model,
            "transformer": transformer,
            "feature_names": feature_names,
            "problem_type": problem_type,
            "target_column": target_column,
            "user_id": user_id,
            "dataset_id": dataset_id,
            "expires_at": time.time() + ttl_seconds,
        }
    return model_id


def _get_owned_model(model_id: str, user_id: str) -> dict[str, Any] | None:
    """Returns the registry entry only if it exists, hasn't expired, AND
    belongs to this user - None in every other case (expired, unknown id,
    or someone else's model), so a caller can turn None into a clean 404
    without ever revealing whether a model_id exists for a different user."""
    with _model_registry_lock:
        _evict_expired_models_locked()
        entry = _model_registry.get(model_id)
        if entry is None or entry["user_id"] != user_id:
            return None
        return entry


def predict_with_model(model_id: str, user_id: str, rows: list[dict]) -> dict:
    """Applies the EXACT preprocessing fitted during training (same
    imputer medians, same scaler mean/std, same one-hot categories) to
    brand new rows, then predicts with the model that was actually chosen
    as best - this is what makes AutoML a usable model rather than a
    one-time report."""
    entry = _get_owned_model(model_id, user_id)
    if entry is None:
        raise ValueError(
            "This trained model wasn't found - it may have expired "
            f"(trained models are kept for {MODEL_REGISTRY_TTL_SECONDS // 60} minutes) "
            "or belongs to a different session. Re-run AutoML to train a new one."
        )

    if not rows:
        raise ValueError("At least one row is required to predict on.")

    input_df = pd.DataFrame(rows)
    transformer: FeatureTransformer = entry["transformer"]
    model = entry["model"]

    X = transformer.transform(input_df)
    raw_predictions = model.predict(X)
    predictions = transformer.decode_target(raw_predictions)

    response: dict[str, Any] = {"predictions": predictions, "model_id": model_id, "n_rows": len(rows)}

    if entry["problem_type"] == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = (
            transformer.target_encoder.classes_.tolist()
            if transformer.target_encoder is not None
            else [str(c) for c in model.classes_]
        )
        response["probabilities"] = [dict(zip(classes, row.tolist())) for row in probabilities]

    return response


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
    warnings_block = (
        "\n\nDATA QUALITY NOTES (mention these plainly if relevant, don't ignore them):\n"
        + "\n".join(f"- {w}" for w in result.warnings)
        if result.warnings
        else ""
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
{warnings_block}

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
