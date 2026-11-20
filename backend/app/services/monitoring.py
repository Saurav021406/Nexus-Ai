"""Persistence and polling for user-owned Supabase KPI monitors."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.supabase_client import supabase_admin

logger = logging.getLogger(__name__)
RULES_TABLE = "monitoring_rules"
ALERTS_TABLE = "monitoring_alerts"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
POLL_INTERVAL_SECONDS = 300


class MonitoringStorageError(RuntimeError):
    """Raised when the monitoring tables are unavailable or a write fails."""


def _validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a simple PostgreSQL table or column name.")
    return value


def read_metric_value(table_name: str, metric_column: str, user_id: str) -> float:
    table_name = _validate_identifier(table_name, "Table name")
    metric_column = _validate_identifier(metric_column, "Metric column")

    try:
        result = (
            supabase_admin.table(table_name)
            .select(f"user_id,{metric_column}")
            .eq("user_id", user_id)
            .limit(2)
            .execute()
        )
    except Exception as error:
        error_code = getattr(error, "code", None)
        if error_code == "PGRST205":
            raise ValueError(
                f"Supabase cannot find public.{table_name} in its API schema cache. "
                "Confirm the source table/view exists in this project, then run "
                "NOTIFY pgrst, 'reload schema' in the Supabase SQL Editor."
            ) from error
        if error_code == "PGRST204":
            raise ValueError(
                f"Supabase cannot find column '{metric_column}' on public.{table_name}. "
                "Check the table/view column name and reload the API schema if it was just created."
            ) from error
        logger.exception(
            "Could not read monitor source public.%s, metric column %s",
            table_name,
            metric_column,
        )
        raise ValueError(
            "Could not read this source. Use a public Supabase table or view with "
            "a user_id column and the selected metric column."
        ) from error

    rows = result.data or []
    if not rows:
        raise ValueError("No row for your user_id was found in that table or view.")
    if len(rows) != 1:
        raise ValueError(
            "The source must return exactly one row per user. Create a KPI view that aggregates "
            "your data to one current value per user."
        )

    value = rows[0].get(metric_column)
    if isinstance(value, bool):
        raise ValueError("The selected metric must be numeric.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("The selected metric value must be numeric.") from error
    if not math.isfinite(numeric_value):
        raise ValueError("The selected metric value must be a finite number.")
    return numeric_value


def _execute_storage(operation):
    try:
        return operation()
    except Exception as error:
        raise MonitoringStorageError(
            "Monitoring storage is unavailable. Run backend/monitoring_tables.sql "
            "in the Supabase SQL Editor, then try again."
        ) from error


def list_rules(user_id: str) -> list[dict[str, Any]]:
    result = _execute_storage(
        lambda: (
            supabase_admin.table(RULES_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
    )
    return result.data or []


def create_rule(user_id: str, rule: dict[str, Any]) -> dict[str, Any]:
    row = {**rule, "user_id": user_id, "is_active": True, "is_triggered": False}
    result = _execute_storage(
        lambda: supabase_admin.table(RULES_TABLE).insert(row).execute()
    )
    if not result.data:
        raise MonitoringStorageError("Monitoring rule was not saved.")
    return result.data[0]


def delete_rule(rule_id: str, user_id: str) -> bool:
    result = _execute_storage(
        lambda: (
            supabase_admin.table(RULES_TABLE)
            .delete()
            .eq("id", rule_id)
            .eq("user_id", user_id)
            .execute()
        )
    )
    return bool(result.data)


def list_alerts(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    result = _execute_storage(
        lambda: (
            supabase_admin.table(ALERTS_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    )
    return result.data or []


def acknowledge_alert(alert_id: str, user_id: str) -> dict[str, Any] | None:
    result = _execute_storage(
        lambda: (
            supabase_admin.table(ALERTS_TABLE)
            .update({"acknowledged_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", alert_id)
            .eq("user_id", user_id)
            .execute()
        )
    )
    return result.data[0] if result.data else None


def _send_alert_email(user_id: str, rule_name: str, message_text: str) -> None:
    user_response = supabase_admin.auth.admin.get_user_by_id(user_id)
    auth_user = getattr(user_response, "user", None)
    recipient = getattr(auth_user, "email", None)
    if not recipient:
        raise ValueError("The account has no email address for alert delivery.")

    message = EmailMessage()
    message["From"] = settings.smtp_from_email
    message["To"] = recipient
    message["Subject"] = f"Nexus AI alert: {rule_name}"
    message.set_content(message_text)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as client:
        client.ehlo()
        client.starttls(context=ssl.create_default_context())
        client.ehlo()
        client.login(settings.smtp_username, settings.smtp_password)
        client.send_message(message)


def evaluate_rule(rule: dict[str, Any], value: float | None = None) -> dict[str, Any]:
    if value is None:
        value = read_metric_value(rule["source_table"], rule["metric_column"], rule["user_id"])

    threshold = float(rule["threshold"])
    triggered = value >= threshold if rule["comparison"] == "above" else value <= threshold
    was_triggered = bool(rule.get("is_triggered"))
    now = datetime.now(timezone.utc).isoformat()

    if triggered and not was_triggered:
        alert = {
            "rule_id": rule["id"],
            "user_id": rule["user_id"],
            "rule_name": rule["name"],
            "metric_value": value,
            "threshold": threshold,
            "comparison": rule["comparison"],
            "message": (
                f"{rule['name']}: value {value:g} is "
                f"{'at or above' if rule['comparison'] == 'above' else 'at or below'} "
                f"the threshold {threshold:g}."
            ),
            "email_status": "pending" if rule.get("email_enabled") else "not_requested",
        }
        inserted_alerts = _execute_storage(
            lambda: supabase_admin.table(ALERTS_TABLE).insert(alert).execute()
        ).data or []
        if not inserted_alerts:
            raise MonitoringStorageError("Monitoring alert was not saved.")

    update = {
        "last_value": value,
        "last_checked_at": now,
        "is_triggered": triggered,
    }
    _execute_storage(
        lambda: (
            supabase_admin.table(RULES_TABLE)
            .update(update)
            .eq("id", rule["id"])
            .eq("user_id", rule["user_id"])
            .execute()
        )
    )

    if triggered and not was_triggered and rule.get("email_enabled"):
        alert_id = inserted_alerts[0].get("id")
        if not alert_id:
            raise MonitoringStorageError("Saved alert did not return an ID for email delivery tracking.")
        try:
            _send_alert_email(rule["user_id"], rule["name"], alert["message"])
        except Exception:
            email_status = "failed"
            logger.exception("Could not deliver monitoring alert email for rule %s", rule.get("id"))
        else:
            email_status = "sent"

        _execute_storage(
            lambda: (
                supabase_admin.table(ALERTS_TABLE)
                .update({"email_status": email_status})
                .eq("id", alert_id)
                .eq("user_id", rule["user_id"])
                .execute()
            )
        )

    return {**rule, **update}


def run_monitoring_cycle() -> None:
    try:
        rules = _execute_storage(
            lambda: (
                supabase_admin.table(RULES_TABLE)
                .select("*")
                .eq("is_active", True)
                .execute()
            )
        ).data or []
    except MonitoringStorageError:
        logger.exception("Could not load monitoring rules")
        return

    for rule in rules:
        try:
            evaluate_rule(rule)
        except (ValueError, MonitoringStorageError):
            logger.exception("Could not evaluate monitoring rule %s", rule.get("id"))
        except Exception:
            logger.exception("Unexpected error evaluating monitoring rule %s", rule.get("id"))


async def monitoring_loop() -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        await asyncio.to_thread(run_monitoring_cycle)
