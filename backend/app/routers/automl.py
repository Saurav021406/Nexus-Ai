"""AutoML endpoints (Phase 5 of the roadmap).

    POST /automl/run     -> real model training + CV + comparison + SHAP +
                             plain-English Business Analyst summary +
                             a model_id that can be used with /automl/predict
    POST /automl/predict -> real predictions from a previously trained
                             model, using the exact preprocessing it was
                             trained with (see services/automl.py's
                             FeatureTransformer)
    POST /automl/cluster -> real KMeans clustering with automatic k selection
    POST /automl/predict/csv -> batch predictions from an uploaded CSV,
                             downloaded back as a CSV with prediction
                             (and probability) columns appended

Both /run and /cluster are synchronous, single-request endpoints (same
pattern as /agent/run) - model training on a typical dataset-studio-sized
CSV finishes in well under the request timeout, so there's no need for the
streaming/polling machinery /agent/run/stream uses for the much longer
multi-agent workflows. /predict is fast enough to always be synchronous -
it's just applying an already-fitted pipeline to a handful of new rows.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd

from app.deps import get_current_user
from app.services import approvals, automl
from app.services.datasets import get_dataset_dataframe
from app.services.rate_limiter import rate_limit

router = APIRouter(prefix="/automl", tags=["automl"])

RESOURCE_TYPE = "automl_run"


class AutoMLRunRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: str | None = None  # "classification" | "regression" | None (auto-detect)


class AutoMLPredictRequest(BaseModel):
    model_id: str
    rows: list[dict]
    explain: bool = False  # per-row SHAP breakdown of THIS prediction, not just global importance


class AutoMLClusterRequest(BaseModel):
    dataset_id: str
    feature_columns: list[str] | None = None  # None = use all numeric columns


class AutoMLAnomalyRequest(BaseModel):
    dataset_id: str
    feature_columns: list[str] | None = None  # None = use all numeric columns
    contamination: float = 0.05  # expected proportion of anomalies


def _run_automl(dataset_id: str, user_id: str, target_column: str, problem_type: str | None) -> dict:
    dataframe = get_dataset_dataframe(dataset_id, user_id)

    try:
        result, fitted_model, X_test, feature_names, transformer = automl.train_and_compare(
            dataframe, target_column, problem_type=problem_type
        )
    except ValueError as e:
        # Bad input (unknown column, too few rows, etc.) - this is the
        # user's mistake to fix, not a server error.
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    shap_importances, shap_unavailable_reason = automl.compute_shap_importance(fitted_model, X_test, feature_names)
    result.shap_importances = shap_importances
    result.shap_unavailable_reason = shap_unavailable_reason

    result.model_id = automl.register_model(
        fitted_model,
        transformer,
        feature_names,
        result.problem_type,
        target_column,
        user_id,
        dataset_id,
        background_sample=X_test[:100],
    )

    business_summary = automl.explain_results(result)

    response = {
        "problem_type": result.problem_type,
        "target_column": result.target_column,
        "feature_columns": result.feature_columns,
        "n_rows_used": result.n_rows_used,
        "n_rows_dropped": result.n_rows_dropped,
        "primary_metric": result.primary_metric,
        "best_model_name": result.best_model_name,
        "model_id": result.model_id,
        "models": [
            {
                "name": m.name,
                "cv_score_mean": m.cv_score_mean,
                "cv_score_std": m.cv_score_std,
                "cv_metric": m.cv_metric,
                "test_metrics": m.test_metrics,
            }
            for m in result.models
        ],
        "shap_importances": result.shap_importances,
        "shap_unavailable_reason": result.shap_unavailable_reason,
        "warnings": result.warnings,
        "excluded_id_columns": result.excluded_id_columns,
        "class_imbalance": result.class_imbalance,
        "business_summary": business_summary,
    }

    # Version history (same Human Approval Hooks table routers/agent.py
    # already uses for Multi-Agent results, and routers/cleaning.py now
    # uses for cleaning runs - see services/approvals.py). Note: model_id
    # is saved for reference, but the actual trained model behind it is
    # in-memory only (see automl.py's MODEL_REGISTRY_TTL_SECONDS) - a
    # historical run's model_id will stop working for /automl/predict
    # after that TTL, even though the metrics/SHAP/summary here stay
    # available forever.
    version = approvals.create_version(
        resource_type=RESOURCE_TYPE,
        resource_id=dataset_id,
        user_id=user_id,
        content=response,
        dataset_id=dataset_id,
    )
    response["version_id"] = version["id"]
    response["version_number"] = version["version_number"]

    return response


@router.get("/versions")
async def list_automl_versions(dataset_id: str, user=Depends(get_current_user)):
    """History of every past AutoML run for this dataset - lets the UI
    show "Run 3 (best: XGBoost, 94% accuracy)" etc. instead of only ever
    exposing the single most recent result."""
    versions = await run_in_threadpool(approvals.list_versions, RESOURCE_TYPE, dataset_id, user.id)
    return {
        "versions": [
            {
                "id": v["id"],
                "version_number": v["version_number"],
                "created_at": v["created_at"],
                "problem_type": v["content"].get("problem_type"),
                "target_column": v["content"].get("target_column"),
                "best_model_name": v["content"].get("best_model_name"),
                "models": v["content"].get("models"),
            }
            for v in versions
        ]
    }


@router.post("/run", dependencies=[Depends(rate_limit("automl_run", max_requests=5, window_seconds=60))])
async def run_automl(payload: AutoMLRunRequest, user=Depends(get_current_user)):
    return await run_in_threadpool(_run_automl, payload.dataset_id, user.id, payload.target_column, payload.problem_type)


@router.post("/predict")
async def predict_automl(payload: AutoMLPredictRequest, user=Depends(get_current_user)):
    def _run():
        try:
            return automl.predict_with_model(payload.model_id, user.id, payload.rows, explain=payload.explain)
        except ValueError as e:
            # Covers both "model not found/expired/not yours" and bad
            # input rows - both are the caller's problem to fix, not a
            # server error, and the message never reveals whether a
            # model_id exists for someone else (see _get_owned_model).
            raise HTTPException(status_code=400, detail=str(e))

    return await run_in_threadpool(_run)


MAX_BATCH_PREDICT_ROWS = 5000


@router.post("/predict/csv")
async def predict_automl_csv(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Batch prediction: upload a CSV of new rows, get back the SAME CSV
    with a `prediction` column (and `probability_<class>` columns for
    classification) appended - much more practical than the single-row
    form for anyone who actually has a batch of new records to score."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for batch prediction.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    def _run() -> tuple[str, bytes]:
        try:
            input_df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read this CSV: {e}")

        if len(input_df) == 0:
            raise HTTPException(status_code=400, detail="This CSV has no rows to predict on.")
        if len(input_df) > MAX_BATCH_PREDICT_ROWS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This CSV has {len(input_df)} rows - batch prediction is limited to "
                    f"{MAX_BATCH_PREDICT_ROWS} rows at a time."
                ),
            )

        try:
            prediction_response = automl.predict_with_model(
                model_id, user.id, input_df.to_dict(orient="records")
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        output_df = input_df.copy()
        output_df["prediction"] = prediction_response["predictions"]
        if "probabilities" in prediction_response:
            prob_df = pd.DataFrame(prediction_response["probabilities"]).reset_index(drop=True)
            prob_df.columns = [f"probability_{c}" for c in prob_df.columns]
            output_df = pd.concat([output_df.reset_index(drop=True), prob_df], axis=1)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        base_name = file.filename.rsplit(".", 1)[0] if file.filename else "predictions"
        return f"{base_name}_predictions.csv", csv_bytes

    download_filename, csv_bytes = await run_in_threadpool(_run)

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{download_filename}"'},
    )


@router.post("/cluster")
async def run_clustering(payload: AutoMLClusterRequest, user=Depends(get_current_user)):
    def _run():
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        try:
            return automl.cluster_dataset(dataframe, feature_columns=payload.feature_columns)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return await run_in_threadpool(_run)


@router.post("/anomalies")
async def run_anomaly_detection(payload: AutoMLAnomalyRequest, user=Depends(get_current_user)):
    def _run():
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        try:
            return automl.detect_anomalies(
                dataframe, feature_columns=payload.feature_columns, contamination=payload.contamination
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return await run_in_threadpool(_run)
