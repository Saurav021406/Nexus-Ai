"""Scenario Simulator + What-if Analysis endpoint (2 of the 3 starred
features in the master vision doc, built together since they're the same
underlying mechanism)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.deps import get_current_user
from app.services import scenario_simulator
from app.services.datasets import get_dataset_dataframe

router = APIRouter(prefix="/scenario", tags=["scenario"])


class ScenarioAdjustment(BaseModel):
    column: str
    type: str  # "percent" | "absolute" | "set"
    value: float | str


class ScenarioSimulateRequest(BaseModel):
    dataset_id: str
    model_id: str
    adjustments: list[ScenarioAdjustment]
    sample_size: int = 500


@router.post("/simulate")
async def simulate_scenario(payload: ScenarioSimulateRequest, user=Depends(get_current_user)):
    def _run() -> dict:
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        try:
            result = scenario_simulator.simulate_scenario(
                dataframe,
                payload.model_id,
                user.id,
                adjustments=[a.model_dump() for a in payload.adjustments],
                sample_size=payload.sample_size,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        business_summary = scenario_simulator.explain_scenario(result)

        return {
            "n_rows_simulated": result.n_rows_simulated,
            "adjustments": [
                {"column": a.column, "type": a.type, "value": a.value} for a in result.adjustments
            ],
            "outcome_type": result.outcome_type,
            "baseline_mean": result.baseline_mean,
            "scenario_mean": result.scenario_mean,
            "delta": result.delta,
            "pct_change": result.pct_change,
            "baseline_distribution": result.baseline_distribution,
            "scenario_distribution": result.scenario_distribution,
            "business_summary": business_summary,
        }

    return await run_in_threadpool(_run)
