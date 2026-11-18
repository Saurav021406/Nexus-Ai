"""Append-only audit events for authenticated write actions."""

from __future__ import annotations

import logging
from typing import Any

from app.supabase_client import supabase_admin

logger = logging.getLogger(__name__)
TABLE = "audit_logs"


class AuditLogStorageError(RuntimeError):
    """Raised when audit history cannot be queried."""


def record_audit_event(
    user_id: str,
    action: str,
    endpoint: str,
    status_code: int,
    duration_ms: int,
) -> None:
    event = {
        "user_id": user_id,
        "action": action,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": max(0, duration_ms),
    }
    try:
        result = supabase_admin.table(TABLE).insert(event).execute()
        if not result.data:
            logger.error("Audit event insert returned no row for user %s, action %s", user_id, action)
    except Exception:
        logger.exception("Could not write audit event for user %s, action %s", user_id, action)


def list_audit_events(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    try:
        result = (
            supabase_admin.table(TABLE)
            .select("id, action, endpoint, status_code, duration_ms, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as error:
        raise AuditLogStorageError(
            "Audit log storage is unavailable. Run backend/audit_logs_table.sql "
            "in the Supabase SQL Editor, then try again."
        ) from error
    return result.data or []
