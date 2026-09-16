import numpy as np
import pandas as pd
import pytest

from app.services import automl, scenario_simulator


def _build_strong_signal_regression_df(n=300, random_state=0):
    """target is STRONGLY, positively driven by 'price' (target = 5*price
    + small noise) - increasing price in a scenario should reliably and
    substantially increase the predicted target. A real, verifiable
    directional relationship, not just "did it run"."""
    rng = np.random.default_rng(random_state)
    price = rng.uniform(10, 100, size=n)
    other_feature = rng.normal(size=n)
    target = 5 * price + rng.normal(0, 5, size=n)
    return pd.DataFrame({"price": price, "other_feature": other_feature, "target": target})


def _train_and_register(df, target_col="target"):
    result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(df, target_col)
    model_id = automl.register_model(
        fitted_model, transformer, feature_names, result.problem_type, target_col,
        user_id="u1", dataset_id="d1", background_sample=X_test[:50],
    )
    return model_id


def test_increasing_a_strongly_positive_driver_increases_predicted_outcome():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "percent", "value": 50}],
    )

    assert result.outcome_type == "numeric"
    assert result.scenario_mean > result.baseline_mean
    assert result.delta > 0
    assert result.pct_change > 0


def test_decreasing_a_strongly_positive_driver_decreases_predicted_outcome():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "percent", "value": -50}],
    )

    assert result.scenario_mean < result.baseline_mean
    assert result.delta < 0


def test_absolute_adjustment_type_works():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "absolute", "value": 20}],
    )

    assert result.delta > 50


def test_irrelevant_feature_adjustment_has_much_smaller_effect_than_the_real_driver():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    price_result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "percent", "value": 50}],
    )
    noise_result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "other_feature", "type": "absolute", "value": 5}],
    )

    assert abs(price_result.delta) > abs(noise_result.delta)


def test_multiple_simultaneous_adjustments():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[
            {"column": "price", "type": "percent", "value": 20},
            {"column": "other_feature", "type": "absolute", "value": 1},
        ],
    )
    assert len(result.adjustments) == 2


def test_classification_outcome_returns_distribution_shift():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "f0", "type": "absolute", "value": 3}],
    )

    assert result.outcome_type == "categorical"
    assert result.baseline_distribution
    assert result.scenario_distribution
    assert round(sum(result.baseline_distribution.values()), 2) == 1.0


def test_set_adjustment_type_on_a_numeric_column():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "set", "value": 100}],
    )
    assert result.outcome_type == "numeric"


def test_raises_on_unknown_column():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)
    with pytest.raises(ValueError):
        scenario_simulator.simulate_scenario(
            df.drop(columns=["target"]), model_id, "u1",
            adjustments=[{"column": "does_not_exist", "type": "percent", "value": 10}],
        )


def test_raises_on_percent_adjustment_to_non_numeric_column():
    df = _build_strong_signal_regression_df()
    df["category"] = "A"
    model_id = _train_and_register(df)
    with pytest.raises(ValueError):
        scenario_simulator.simulate_scenario(
            df.drop(columns=["target"]), model_id, "u1",
            adjustments=[{"column": "category", "type": "percent", "value": 10}],
        )


def test_raises_on_invalid_adjustment_type():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)
    with pytest.raises(ValueError):
        scenario_simulator.simulate_scenario(
            df.drop(columns=["target"]), model_id, "u1",
            adjustments=[{"column": "price", "type": "multiply", "value": 2}],
        )


def test_raises_on_no_adjustments():
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)
    with pytest.raises(ValueError):
        scenario_simulator.simulate_scenario(df.drop(columns=["target"]), model_id, "u1", adjustments=[])


def test_raises_on_unknown_model_via_predict_with_model():
    df = _build_strong_signal_regression_df()
    with pytest.raises(ValueError):
        scenario_simulator.simulate_scenario(
            df.drop(columns=["target"]), "does-not-exist", "u1",
            adjustments=[{"column": "price", "type": "percent", "value": 10}],
        )


def test_sample_size_caps_the_number_of_rows_simulated():
    df = _build_strong_signal_regression_df(n=300)
    model_id = _train_and_register(df)

    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "percent", "value": 10}],
        sample_size=50,
    )
    assert result.n_rows_simulated == 50


def test_explain_scenario_grounds_prompt_in_exact_numbers(monkeypatch):
    df = _build_strong_signal_regression_df()
    model_id = _train_and_register(df)
    result = scenario_simulator.simulate_scenario(
        df.drop(columns=["target"]), model_id, "u1",
        adjustments=[{"column": "price", "type": "percent", "value": 50}],
    )

    captured = {}

    def fake_consensus_json(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"summary": "fake", "key_metrics": ["fake"], "recommendation": "fake"}

    monkeypatch.setattr(scenario_simulator, "get_consensus_json", fake_consensus_json)

    output = scenario_simulator.explain_scenario(result)

    assert output["summary"] == "fake"
    assert str(result.baseline_mean) in captured["prompt"]
    assert "price" in captured["prompt"]
