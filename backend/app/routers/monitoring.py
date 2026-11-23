"""Live KPI monitor and in-app alert endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import get_current_user
from app.services import monitoring

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class MonitorPreviewRequest(BaseModel):
    source_table: str = Field(min_length=1, max_length=63)
    metric_column: str = Field(min_length=1, max_length=63)


class MonitorCreateRequest(MonitorPreviewRequest):
    name: str = Field(min_length=1, max_length=100)
    comparison: Literal["above", "below"]
    threshold: float = Field(allow_inf_nan=False)
    email_enabled: bool = False


def _raise_monitoring_error(error: Exception) -> None:
    if isinstance(error, monitoring.MonitoringStorageError):
        raise HTTPException(status_code=503, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    raise error


@router.get("/rules")
async def get_monitor_rules(user=Depends(get_current_user)):
    try:
        rules = await run_in_threadpool(monitoring.list_rules, user.id)
    except monitoring.MonitoringStorageError as error:
        _raise_monitoring_error(error)
    return {"rules": rules, "poll_interval_seconds": monitoring.POLL_INTERVAL_SECONDS}


@router.post("/preview")
async def preview_monitor_source(payload: MonitorPreviewRequest, user=Depends(get_current_user)):
    try:
        value = await run_in_threadpool(
            monitoring.read_metric_value,
            payload.source_table,
            payload.metric_column,
            user.id,
        )
    except ValueError as error:
        _raise_monitoring_error(error)
    return {"current_value": value}


@router.post("/rules")
async def create_monitor_rule(payload: MonitorCreateRequest, user=Depends(get_current_user)):
    if payload.email_enabled and not settings.smtp_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Email alerts are not configured. Set SMTP_HOST, SMTP_USERNAME, "
                "SMTP_PASSWORD, and SMTP_FROM_EMAIL on the backend."
            ),
        )
    try:
        current_value = await run_in_threadpool(
            monitoring.read_metric_value,
            payload.source_table,
            payload.metric_column,
            user.id,
        )
        rule = await run_in_threadpool(
            monitoring.create_rule,
            user.id,
            payload.model_dump(exclude={"source_table", "metric_column"})
            | {
                "source_table": payload.source_table,
                "metric_column": payload.metric_column,
                "last_value": current_value,
                "last_checked_at": None,
            },
        )
        rule = await run_in_threadpool(monitoring.evaluate_rule, rule, current_value)
    except (ValueError, monitoring.MonitoringStorageError) as error:
        _raise_monitoring_error(error)
    return {"rule": rule}


@router.delete("/rules/{rule_id}")
async def remove_monitor_rule(rule_id: str, user=Depends(get_current_user)):
    try:
        deleted = await run_in_threadpool(monitoring.delete_rule, rule_id, user.id)
    except monitoring.MonitoringStorageError as error:
        _raise_monitoring_error(error)
    if not deleted:
        raise HTTPException(status_code=404, detail="Monitor not found")
    return {"deleted": True, "rule_id": rule_id}


@router.get("/alerts")
async def get_monitor_alerts(user=Depends(get_current_user)):
    try:
        alerts = await run_in_threadpool(monitoring.list_alerts, user.id)
    except monitoring.MonitoringStorageError as error:
        _raise_monitoring_error(error)
    return {"alerts": alerts}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_monitor_alert(alert_id: str, user=Depends(get_current_user)):
    try:
        alert = await run_in_threadpool(monitoring.acknowledge_alert, alert_id, user.id)
    except monitoring.MonitoringStorageError as error:
        _raise_monitoring_error(error)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"alert": alert}
