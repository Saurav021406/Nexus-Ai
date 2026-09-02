import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from app.services import automl


def _classification_df(n_samples=200, n_features=5, random_state=0):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=3, n_redundant=0, random_state=random_state
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df, "target"


# --- ID-like column exclusion ------------------------------------------

def test_id_like_string_column_is_excluded_from_training():
    df, target_col = _classification_df(n_samples=100)
    df["customer_id"] = [f"cust-{i}" for i in range(100)]  # unique per row, id-like name

    result, _, _, feature_names, _ = automl.train_and_compare(df, target_col)

    assert "customer_id" in result.excluded_id_columns
    assert not any("customer_id" in f for f in feature_names)


def test_unique_numeric_column_without_id_like_name_is_kept():
    # A genuinely continuous, all-unique numeric feature should NOT be
    # treated as an id just because it happens to be unique - only
    # non-numeric or id-named unique columns are excluded.
    rng = np.random.default_rng(0)
    df, target_col = _classification_df(n_samples=100)
    df["precise_measurement"] = rng.normal(size=100)  # all-unique floats, ordinary name

    result, _, _, feature_names, _ = automl.train_and_compare(df, target_col)

    assert "precise_measurement" not in result.excluded_id_columns
    assert "precise_measurement" in feature_names


def test_sequential_integer_id_column_is_excluded():
    df, target_col = _classification_df(n_samples=100)
    df["row_id"] = range(100)  # unique, id-like name, numeric

    result, _, _, _, _ = automl.train_and_compare(df, target_col)

    assert "row_id" in result.excluded_id_columns


# --- Leakage warnings -----------------------------------------------------

def test_feature_perfectly_correlated_with_target_triggers_leakage_warning():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    df = pd.DataFrame({
        "leaky_feature": y.astype(float) + rng.normal(0, 0.001, size=100),  # near-perfect proxy for y
        "normal_feature": rng.normal(size=100),
        "target": y,
    })

    result, _, _, _, _ = automl.train_and_compare(df, "target")

    assert any("leaky_feature" in w for w in result.warnings)


def test_normal_features_do_not_trigger_leakage_warning():
    df, target_col = _classification_df(n_samples=200, n_features=5)
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    assert not any("correlated with the target" in w for w in result.warnings)


# --- Class imbalance --------------------------------------------------------

def test_severely_imbalanced_target_is_detected_and_flagged():
    rng = np.random.default_rng(0)
    n = 300
    # ~95/5 split
    y = np.array([0] * 285 + [1] * 15)
    rng.shuffle(y)
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n) + y * 2,  # give the minority class a learnable signal
        "target": y,
    })

    result, fitted_model, _, _, _ = automl.train_and_compare(df, "target")

    assert result.class_imbalance is not None
    assert result.class_imbalance["detected"] is True
    assert result.class_imbalance["ratio"] >= automl.CLASS_IMBALANCE_RATIO_THRESHOLD
    assert any("imbalanced" in w for w in result.warnings)
    # the best model should have been trained with balancing applied -
    # check the actual fitted estimator's constructor argument where
    # available (LogisticRegression/RandomForest expose class_weight)
    if hasattr(fitted_model, "class_weight"):
        assert fitted_model.class_weight == "balanced"


def test_balanced_target_is_not_flagged():
    df, target_col = _classification_df(n_samples=200)  # make_classification defaults to balanced classes
    result, _, _, _, _ = automl.train_and_compare(df, target_col)
    assert result.class_imbalance is None


# --- Non-numeric classification target + prediction decoding ---------------

def _string_label_df(n_samples=200):
    X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = np.where(y == 1, "yes", "no")
    return df


def test_string_classification_target_is_encoded_and_decoded_correctly():
    df = _string_label_df()
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, "target")

    assert transformer.target_encoder is not None
    raw_preds = fitted_model.predict(X_test)
    decoded = transformer.decode_target(raw_preds)
    assert set(decoded) <= {"yes", "no"}


# --- Prediction endpoint (service-level, real train -> register -> predict) --

def test_predict_with_model_returns_real_predictions_for_new_rows():
    df, target_col = _classification_df(n_samples=200, n_features=4)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)

    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, target_col, user_id="u1", dataset_id="d1"
    )

    new_rows = df.drop(columns=[target_col]).head(3).to_dict(orient="records")
    response = automl.predict_with_model(model_id, "u1", new_rows)

    assert len(response["predictions"]) == 3
    assert response["model_id"] == model_id
    assert "probabilities" in response  # classification + predict_proba available


def test_predict_with_regression_model_has_no_probabilities():
    X, y = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, "target")
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, "target", user_id="u1", dataset_id="d1"
    )

    new_rows = df.drop(columns=["target"]).head(2).to_dict(orient="records")
    response = automl.predict_with_model(model_id, "u1", new_rows)

    assert len(response["predictions"]) == 2
    assert "probabilities" not in response


def test_predict_with_wrong_user_id_is_rejected():
    df, target_col = _classification_df(n_samples=200)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, target_col, user_id="owner", dataset_id="d1"
    )

    new_rows = df.drop(columns=[target_col]).head(1).to_dict(orient="records")
    with pytest.raises(ValueError):
        automl.predict_with_model(model_id, "someone-else", new_rows)


def test_predict_with_unknown_model_id_is_rejected():
    with pytest.raises(ValueError):
        automl.predict_with_model("does-not-exist", "u1", [{"f0": 1.0}])


def test_predict_with_no_rows_is_rejected():
    df, target_col = _classification_df(n_samples=200)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, target_col, user_id="u1", dataset_id="d1"
    )
    with pytest.raises(ValueError):
        automl.predict_with_model(model_id, "u1", [])


def test_predict_handles_a_row_missing_some_feature_columns():
    # A new row missing a column entirely should still work - the
    # transformer reindexes to the trained feature set and imputes the
    # missing value, rather than crashing.
    df, target_col = _classification_df(n_samples=200, n_features=4)
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, target_col, user_id="u1", dataset_id="d1"
    )

    partial_row = {feature_names[0]: 0.5}  # only the first feature provided
    response = automl.predict_with_model(model_id, "u1", [partial_row])
    assert len(response["predictions"]) == 1
