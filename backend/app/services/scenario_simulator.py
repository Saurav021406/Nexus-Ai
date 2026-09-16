"""Scenario Simulator + What-if Analysis (2 of the 3 starred ⭐⭐⭐ features
in the master vision doc - built together since they're the same
mechanism: "if we change X, what happens to Y" using a model that's
already been trained).

    trained model (from AutoML) + a set of feature adjustments
        -> baseline: predict on the CURRENT rows, unmodified
        -> scenario: apply the adjustment(s) to a COPY of those rows,
                     predict again
        -> compare the two outcomes (mean shift for a numeric target,
           class distribution shift for a categorical one)
        -> explain_scenario(): Business Analyst LLM layer - plain
           English, grounded ONLY in the exact numbers above

Deliberately built on TOP of services/automl.py's existing PUBLIC API
(predict_with_model) rather than reaching into its internals - a
perturbed row is predicted exactly the same way a brand new row would be,
through the exact same fitted preprocessing pipeline, so "what happens if
price goes up 10%" is genuinely using the real trained model, not a
separate approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.services import automl
from app.services.consensus import get_consensus_json

DEFAULT_SAMPLE_SIZE = 500
VALID_ADJUSTMENT_TYPES = ("percent", "absolute", "set")


@dataclass
class Adjustment:
    column: str
    type: str
    value: Any


@dataclass
class ScenarioResult:
    n_rows_simulated: int
    adjustments: list[Adjustment]
    outcome_type: str  # "numeric" | "categorical"
    baseline_mean: float | None = None
    scenario_mean: float | None = None
    delta: float | None = None
    pct_change: float | None = None
    baseline_distribution: dict[str, float] = field(default_factory=dict)
    scenario_distribution: dict[str, float] = field(default_factory=dict)


def _apply_adjustment(df: pd.DataFrame, adjustment: dict[str, Any]) -> None:
    """Mutates df in place - caller is responsible for passing a copy,
    never the original dataframe (see simulate_scenario)."""
    column = adjustment.get("column")
    adj_type = adjustment.get("type")
    value = adjustment.get("value")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not in this dataset.")
    if adj_type not in VALID_ADJUSTMENT_TYPES:
        raise ValueError(f"Unknown adjustment type '{adj_type}' for column '{column}' - use 'percent', 'absolute', or 'set'.")

    is_numeric = pd.api.types.is_numeric_dtype(df[column])

    if adj_type in ("percent", "absolute") and not is_numeric:
        raise ValueError(f"'{column}' is not numeric - 'percent'/'absolute' adjustments only work on numeric columns. Use 'set' for categorical columns.")

    if adj_type == "percent":
        if not isinstance(value, (int, float)):
            raise ValueError(f"'percent' adjustment for '{column}' needs a numeric value (e.g. 10 for +10%).")
        df[column] = df[column] * (1 + value / 100)
    elif adj_type == "absolute":
        if not isinstance(value, (int, float)):
            raise ValueError(f"'absolute' adjustment for '{column}' needs a numeric value.")
        df[column] = df[column] + value
    else:  # "set"
        df[column] = value


def simulate_scenario(
    df: pd.DataFrame,
    model_id: str,
    user_id: str,
    adjustments: list[dict[str, Any]],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> ScenarioResult:
    """Simulates "what happens if we change X" by predicting the SAME rows
    twice - once as-is (baseline), once with the requested adjustment(s)
    applied (scenario) - through the actual trained model, via
    automl.predict_with_model(). Both predictions go through the model's
    real, already-fitted preprocessing, so this reflects what the model
    genuinely learned, not a hand-rolled approximation of its behavior.

    sample_size caps how many rows get simulated (predicting the entire
    dataset for every scenario could be slow on a large one) - a random
    sample is representative enough for an aggregate "what happens on
    average" answer, which is what a scenario simulation is actually
    asking.
    """
    if not adjustments:
        raise ValueError("At least one adjustment is required to run a scenario.")

    for adjustment in adjustments:
        if adjustment.get("column") not in df.columns:
            raise ValueError(f"Column '{adjustment.get('column')}' is not in this dataset.")

    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df.copy()

    baseline_rows = sample_df.to_dict(orient="records")
    baseline_response = automl.predict_with_model(model_id, user_id, baseline_rows)

    scenario_df = sample_df.copy()
    applied: list[Adjustment] = []
    for adjustment in adjustments:
        _apply_adjustment(scenario_df, adjustment)
        applied.append(Adjustment(column=adjustment["column"], type=adjustment["type"], value=adjustment["value"]))

    scenario_rows = scenario_df.to_dict(orient="records")
    scenario_response = automl.predict_with_model(model_id, user_id, scenario_rows)

    baseline_predictions = baseline_response["predictions"]
    scenario_predictions = scenario_response["predictions"]

    if all(isinstance(p, (int, float)) for p in baseline_predictions):
        baseline_mean = float(pd.Series(baseline_predictions).mean())
        scenario_mean = float(pd.Series(scenario_predictions).mean())
        delta = scenario_mean - baseline_mean
        pct_change = round((delta / baseline_mean) * 100, 2) if baseline_mean != 0 else None
        return ScenarioResult(
            n_rows_simulated=len(sample_df),
            adjustments=applied,
            outcome_type="numeric",
            baseline_mean=round(baseline_mean, 4),
            scenario_mean=round(scenario_mean, 4),
            delta=round(delta, 4),
            pct_change=pct_change,
        )

    baseline_distribution = pd.Series(baseline_predictions).value_counts(normalize=True).round(4).to_dict()
    scenario_distribution = pd.Series(scenario_predictions).value_counts(normalize=True).round(4).to_dict()
    return ScenarioResult(
        n_rows_simulated=len(sample_df),
        adjustments=applied,
        outcome_type="categorical",
        baseline_distribution={str(k): v for k, v in baseline_distribution.items()},
        scenario_distribution={str(k): v for k, v in scenario_distribution.items()},
    )


def explain_scenario(result: ScenarioResult) -> dict:
    """The Business Analyst layer: plain-English narrative grounded ONLY
    in the exact numbers already computed above - same discipline every
    other specialist in this codebase follows."""
    adjustments_summary = ", ".join(
        f"{a.column} {'+' if a.type == 'percent' and a.value >= 0 else ''}{a.value}"
        f"{'%' if a.type == 'percent' else ''}" + (f" (set to {a.value})" if a.type == "set" else "")
        for a in result.adjustments
    )

    if result.outcome_type == "numeric":
        outcome_summary = (
            f"Predicted outcome changed from {result.baseline_mean} (baseline, average across "
            f"{result.n_rows_simulated} rows) to {result.scenario_mean} under this scenario - "
            f"a change of {result.delta} ({result.pct_change}%)."
        )
    else:
        outcome_summary = (
            f"Predicted outcome distribution shifted from {result.baseline_distribution} (baseline) "
            f"to {result.scenario_distribution} (scenario), across {result.n_rows_simulated} rows."
        )

    prompt = f"""You are a Business Analyst explaining the predicted impact of a hypothetical change,
using ONLY the exact numbers below - never invent or estimate a number that isn't present here.

Scenario adjustment(s): {adjustments_summary}
{outcome_summary}

Write a plain-English explanation of what this scenario means in practice, and whether the predicted
impact is large or small relative to the baseline.

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph explaining the predicted impact of this scenario using only exact numbers above",
  "key_metrics": ["plain-English restatement of the outcome 1", "detail 2"],
  "recommendation": "one concrete next step (e.g. test this change on a subset first, monitor X after change)"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
