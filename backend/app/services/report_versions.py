"""Persistence layer for report versions (Human Approval Hooks).

Every call to /report/generate or /report/{id}/resubmit creates a NEW row
here rather than overwriting anything, so the full approve/reject/regenerate
history for a dataset is always available for the version-history panel.

Note: report content includes embedded chart images as base64 strings, so
each version row can be a few hundred KB. That's an acceptable tradeoff for
this feature's scope (approval workflow), but worth knowing if version count
per dataset grows large - a future optimization could move chart images to
Supabase Storage and store only a reference here.
"""

from typing import Any

from app.supabase_client import supabase_admin

TABLE = "report_versions"


def get_next_version_number(dataset_id: str, user_id: str) -> int:
    result = (
        supabase_admin.table(TABLE)
        .select("version_number")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["version_number"] + 1
    return 1


def create_version(dataset_id: str, user_id: str, content: dict[str, Any]) -> dict[str, Any]:
    version_number = get_next_version_number(dataset_id, user_id)
    row = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "version_number": version_number,
        "content": content,
        "approval_status": "pending",
        "rejection_reason": None,
    }
    result = supabase_admin.table(TABLE).insert(row).execute()
    if not result.data:
        raise RuntimeError("Failed to save report version")
    return result.data[0]


def get_version(report_id: str, user_id: str) -> dict[str, Any] | None:
    result = (
        supabase_admin.table(TABLE)
        .select("*")
        .eq("id", report_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return result.data


def set_status(
    report_id: str,
    user_id: str,
    status: str,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {"approval_status": status}
    update["rejection_reason"] = rejection_reason if status == "rejected" else None

    result = (
        supabase_admin.table(TABLE)
        .update(update)
        .eq("id", report_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        raise RuntimeError("Report version not found or could not be updated")
    return result.data[0]


def list_versions(dataset_id: str, user_id: str) -> list[dict[str, Any]]:
    result = (
        supabase_admin.table(TABLE)
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("user_id", user_id)
        .order("version_number", desc=True)
        .execute()
    )
    return result.data or []
