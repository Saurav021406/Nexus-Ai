"""ML Engineer Agent (Section 16 of the Phase 4 spec) - now wired to the
real AutoML engine (services/automl.py, Phase 5).

Previously this agent only REASONED about modeling readiness via an LLM
and never trained anything. This version does what the roadmap always
described: "automatically decides classification/regression/clustering,
then trains models automatically" - genuine scikit-learn/XGBoost/LightGBM
training, real cross-validation, real SHAP, no mock/demo output at any
step.

Unlike most specialists, this one needs the real dataframe (not just the
privacy-filtered data_summary text) to actually train anything, so it's
registered with needs_full_access=True and receives the full
WorkflowState - the same mechanism agents/sql_agent.py already uses for
the same reason (see that file's docstring, and manager_v2.py's
run_specialist_task for the branch that makes this call).

    task_description (natural language, e.g. "predict customer churn")
        -> _resolve_target_column()      (LLM picks the target from the
                                           REAL column names - the answer
                                           is strictly validated against
                                           them, never trusted blindly)
        -> automl.train_and_compare()    (real training + CV + comparison)
        -> automl.compute_shap_importance()
        -> automl.register_model()       (so the trained model is just as
                                           usable afterward via
                                           /automl/predict as one started
                                           from the dedicated AutoML tab)
        -> automl.explain_results()      (Business Analyst LLM layer -
                                           plain English, grounded only in
                                           the exact numbers above)
        -> approvals.create_version()    (same run history the AutoML tab
                                           already writes to - a Multi-
                                           Agent-triggered run shows up
                                           right alongside manual ones)

Any failure at any step (can't identify a target column, dataset is a
document not a table, too few rows, every candidate model fails to train)
returns a clear {"error": "..."} - agents/executor.py already retries once
and then reports that exact message to the user, so this agent needs no
separate error-reporting path of its own.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.agents.state import WorkflowState
from app.services import approvals, automl
from app.services.consensus import get_consensus_json
from app.services.datasets import get_dataset_dataframe, is_document_dataset


def _resolve_target_column(task_description: str, columns: list[str]) -> str | None:
    """Asks the LLM which column the task is actually asking to predict,
    then STRICTLY validates the answer against the real column list - a
    hallucinated column name can never silently reach automl.train_and_
    compare() this way. That function would raise a clear ValueError on
    an unknown column anyway, but catching it here produces a far more
    useful error message (with the actual available columns listed)."""
    prompt = f"""A user asked a data team to do this: "{task_description}"

The dataset has these exact columns: {columns}

Which ONE column is the user asking to predict, forecast, or classify? Respond with
the EXACT column name from the list above, character-for-character. If the request
isn't actually asking to predict/forecast/classify anything (e.g. it's just asking
for a summary or a chart), respond with null instead.

Respond ONLY in this exact JSON format, no extra text:
{{"target_column": "exact_column_name_or_null"}}"""

    try:
        result = get_consensus_json(prompt, temperature=0, max_tokens=128)
    except Exception:
        return None

    candidate = result.get("target_column")
    if not candidate or candidate not in columns:
        return None
    return candidate


def _serialize_models(models) -> list[dict]:
    return [
        {
            "name": m.name,
            "cv_score_mean": m.cv_score_mean,
            "cv_score_std": m.cv_score_std,
            "cv_metric": m.cv_metric,
            "test_metrics": m.test_metrics,
        }
        for m in models
    ]


def analyze(state: WorkflowState, task_description: str = "") -> dict:
    task_description = task_description or state.user_query or ""

    if is_document_dataset(state.dataset_id, state.user_id):
        return {
            "error": (
                "This is a document (PDF/Word), not a tabular dataset - the ML Engineer needs "
                "columns and rows to train a model on. Try 'Ask your data' for document questions instead."
            )
        }

    try:
        dataframe = get_dataset_dataframe(state.dataset_id, state.user_id)
    except HTTPException as e:
        return {"error": f"Could not load the dataset: {e.detail}"}
    except Exception as e:
        return {"error": f"Could not load the dataset: {e}"}

    target_column = _resolve_target_column(task_description, list(dataframe.columns))
    if not target_column:
        return {
            "error": (
                f'Couldn\'t determine what to predict from "{task_description}" - try naming the '
                f'column directly, e.g. "predict {dataframe.columns[0]}". '
                f"Available columns: {', '.join(dataframe.columns)}"
            )
        }

    try:
        result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(
            dataframe, target_column
        )
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}

    shap_importances, shap_unavailable_reason = automl.compute_shap_importance(
        fitted_model, X_test, feature_names
    )
    result.shap_importances = shap_importances
    result.shap_unavailable_reason = shap_unavailable_reason

    try:
        result.model_id = automl.register_model(
            fitted_model,
            transformer,
            feature_names,
            result.problem_type,
            target_column,
            state.user_id,
            state.dataset_id,
            background_sample=X_test[:100],
        )
    except Exception:
        result.model_id = None  # non-fatal - the analysis result itself is already computed

    business_summary = automl.explain_results(result)

    try:
        approvals.create_version(
            resource_type="automl_run",
            resource_id=state.dataset_id,
            user_id=state.user_id,
            content={
                "problem_type": result.problem_type,
                "target_column": result.target_column,
                "feature_columns": result.feature_columns,
                "n_rows_used": result.n_rows_used,
                "n_rows_dropped": result.n_rows_dropped,
                "primary_metric": result.primary_metric,
                "best_model_name": result.best_model_name,
                "model_id": result.model_id,
                "models": _serialize_models(result.models),
                "shap_importances": result.shap_importances,
                "warnings": result.warnings,
            },
            dataset_id=state.dataset_id,
        )
    except Exception:
        pass  # version history is a bonus, never worth failing the whole task over

    return {
        **business_summary,
        "problem_type": result.problem_type,
        "target_column": result.target_column,
        "best_model_name": result.best_model_name,
        "model_id": result.model_id,
        "models": _serialize_models(result.models),
        "shap_importances": result.shap_importances,
        "shap_unavailable_reason": result.shap_unavailable_reason,
        "warnings": result.warnings,
    }
