"""Canonical workflow state (Section 20 of the Phase 4 spec).

This extends the original WorkflowState (kept 100% backward compatible -
every field the old domain-routing Manager, quality_check, and Report Agent
already read/write is untouched) with the fields a real intent-driven
Manager needs: the actual user_query, a goal, a dependency-aware task list,
tool results, approvals, and a status. Existing code paths
(agents/manager.py, agents/quality.py, routers/domain.py, routers/report.py)
continue to work unmodified - they simply don't set the new fields.

New code (agents/manager_v2.py, routers/agent.py) uses the full shape.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    # --- Original fields (Phase 1-4 domain-routing path). Do not rename or
    # remove: agents/manager.py, agents/quality.py, agents/tracing.py, and
    # routers/domain.py all depend on these exact names. ---
    dataset_id: str
    user_id: str
    data_summary: str
    classification: dict
    include_secondary_specialists: bool = False
    plan: list[dict] = field(default_factory=list)
    specialist_results: dict[str, Any] = field(default_factory=dict)
    review: dict | None = None
    security: dict | None = None
    traces: list[dict] = field(default_factory=list)
    final_output: dict | None = None

    # --- New canonical fields (Section 20). Populated by the new
    # intent-aware Manager (agents/manager_v2.py) and the future Task
    # Planner/Executor. Safe defaults so existing callers that never set
    # these still get a valid, serializable state. ---
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    goal: str = ""
    tasks: list[dict] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    approvals: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    status: str = "running"  # running | awaiting_approval | completed | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def add_error(self, source: str, message: str) -> None:
        self.errors.append({"source": source, "message": message, "timestamp": time.time()})
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Full serializable snapshot - safe to persist to Supabase or
        return over the API (Section 20: "State must be serializable")."""
        return asdict(self)
