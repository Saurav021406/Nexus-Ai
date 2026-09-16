import numpy as np
import pandas as pd
import pytest

from app.services import root_cause


def _build_revenue_drop_dataset() -> pd.DataFrame:
    """60 days of daily revenue across 3 regions. West region revenue
    craters in the second half; East and North stay roughly flat. West
    should be unambiguously identified as the dominant driver of the
    overall drop."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    rows = []
    for i, date in enumerate(dates):
        is_after = i >= 30
        rows.append({"date": date, "region": "West", "revenue": 200 if is_after else 1000})
        rows.append({"date": date, "region": "East", "revenue": 980})
        rows.append({"date": date, "region": "North", "revenue": 1010})
    return pd.DataFrame(rows)


def test_dominant_driver_is_correctly_identified_as_top_contributor():
    df = _build_revenue_drop_dataset()
    result = root_cause.analyze_root_cause(df, metric_column="revenue", date_column="date")

    assert result.overall_change < 0  # a genuine drop
    top = result.top_contributors[0]
    assert top.segment_column == "region"
    assert top.segment_value == "West"
    assert top.delta < 0


def test_stable_segments_are_not_flagged_as_major_contributors():
    df = _build_revenue_drop_dataset()
    result = root_cause.analyze_root_cause(df, metric_column="revenue", date_column="date")

    east = next(c for c in result.top_contributors if c.segment_value == "East")
    assert abs(east.delta) < 5


def test_contribution_percentages_are_directionally_sensible():
    df = _build_revenue_drop_dataset()
    result = root_cause.analyze_root_cause(df, metric_column="revenue", date_column="date")

    west = next(c for c in result.top_contributors if c.segment_value == "West")
    assert west.contribution_pct is not None
    assert west.contribution_pct > 60


def test_direction_increase_sorts_best_contributors_first():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = []
    for i, date in enumerate(dates):
        is_after = i >= 20
        rows.append({"date": date, "product": "Widget", "sales": 500 if is_after else 100})
        rows.append({"date": date, "product": "Gadget", "sales": 300})
    df = pd.DataFrame(rows)

    result = root_cause.analyze_root_cause(df, "sales", "date", direction="increase")

    assert result.overall_change > 0
    top = result.top_contributors[0]
    assert top.segment_value == "Widget"
    assert top.delta > 0


def test_id_like_columns_are_excluded_from_segment_analysis():
    df = _build_revenue_drop_dataset()
    df["transaction_id"] = [f"txn-{i}" for i in range(len(df))]

    result = root_cause.analyze_root_cause(df, "revenue", "date")

    assert "transaction_id" not in result.segment_columns_analyzed


def test_high_cardinality_non_id_columns_are_excluded():
    df = _build_revenue_drop_dataset()
    rng = np.random.default_rng(0)
    df["customer_note"] = [f"note {rng.integers(0, 1000)}" for _ in range(len(df))]

    result = root_cause.analyze_root_cause(df, "revenue", "date")

    assert "customer_note" not in result.segment_columns_analyzed


def test_explicit_segment_columns_are_respected_over_auto_detection():
    df = _build_revenue_drop_dataset()
    df["irrelevant_col"] = "same_value_always"

    result = root_cause.analyze_root_cause(df, "revenue", "date", segment_columns=["region"])

    assert result.segment_columns_analyzed == ["region"]


def test_raises_on_unknown_metric_column():
    df = _build_revenue_drop_dataset()
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "does_not_exist", "date")


def test_raises_on_unknown_date_column():
    df = _build_revenue_drop_dataset()
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "revenue", "does_not_exist")


def test_raises_on_non_numeric_metric_column():
    df = _build_revenue_drop_dataset()
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "region", "date")


def test_raises_on_invalid_aggregation():
    df = _build_revenue_drop_dataset()
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "revenue", "date", aggregation="median")


def test_raises_on_too_few_usable_rows():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "revenue": [100, 200, 150, 175, 190],
    })
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "revenue", "date")


def test_raises_when_all_dates_are_identical():
    df = pd.DataFrame({
        "date": ["2026-01-01"] * 20,
        "revenue": range(20),
    })
    with pytest.raises(ValueError):
        root_cause.analyze_root_cause(df, "revenue", "date")


def test_warns_when_no_segment_columns_are_usable():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "revenue": range(30)})
    result = root_cause.analyze_root_cause(df, "revenue", "date")
    assert len(result.warnings) > 0


def test_explain_root_cause_grounds_prompt_in_exact_numbers(monkeypatch):
    df = _build_revenue_drop_dataset()
    result = root_cause.analyze_root_cause(df, "revenue", "date")

    captured = {}

    def fake_consensus_json(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"summary": "fake", "key_metrics": ["fake"], "recommendation": "fake"}

    monkeypatch.setattr(root_cause, "get_consensus_json", fake_consensus_json)

    output = root_cause.explain_root_cause(result)

    assert output["summary"] == "fake"
    assert "West" in captured["prompt"]
    assert str(result.overall_change) in captured["prompt"]
