"""Generic Human Approval service (Phase 4 roadmap: "Generic Human
Approval, currently only works for reports").

Same version-history + approve/reject/resubmit pattern
services/report_versions.py already uses for reports, generalized with a
resource_type column so ANY kind of AI output - not just reports - can go
through the same approval gate.

Reports keep using report_versions.py completely unchanged (zero
regression risk to a working feature); this is what NEW resource types
plug into, starting with "agent_workflow" (Multi-Agent results - see
routers/agent.py).

Requires the `approvals` table - see backend/sql/approvals_table.sql for
the migration to run in Supabase.
"""

from typing import Any

from app.supabase_client import supabase_admin

TABLE = "approvals"


def get_next_version_number(resource_type: str, resource_id: str, user_id: str) -> int:
    result = (
        supabase_admin.table(TABLE)
        .select("version_number")
        .eq("resource_type", resource_type)
        .eq("resource_id", resource_id)
        .eq("user_id", user_id)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["version_number"] + 1
    return 1


def create_version(
    resource_type: str,
    resource_id: str,
    user_id: str,
    content: dict[str, Any],
    dataset_id: str | None = None,
) -> dict[str, Any]:
    version_number = get_next_version_number(resource_type, resource_id, user_id)
    row = {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "dataset_id": dataset_id,
        "user_id": user_id,
        "version_number": version_number,
        "content": content,
        "approval_status": "pending",
        "rejection_reason": None,
    }
    result = supabase_admin.table(TABLE).insert(row).execute()
    if not result.data:
        raise RuntimeError("Failed to save approval version")
    return result.data[0]


def get_version(approval_id: str, user_id: str) -> dict[str, Any] | None:
    result = (
        supabase_admin.table(TABLE)
        .select("*")
        .eq("id", approval_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return result.data


def set_status(
    approval_id: str,
    user_id: str,
    status: str,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {"approval_status": status}
    update["rejection_reason"] = rejection_reason if status == "rejected" else None

    result = (
        supabase_admin.table(TABLE)
        .update(update)
        .eq("id", approval_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError("Approval not found or could not be updated")
    return result.data[0]


def list_versions(resource_type: str, resource_id: str, user_id: str) -> list[dict[str, Any]]:
    result = (
        supabase_admin.table(TABLE)
        .select("*")
        .eq("resource_type", resource_type)
        .eq("resource_id", resource_id)
        .eq("user_id", user_id)
        .order("version_number", desc=True)
        .execute()
    )
    return result.data or []