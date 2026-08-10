from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
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