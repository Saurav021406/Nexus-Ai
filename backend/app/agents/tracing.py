import time
from typing import Any

from app.agents.state import WorkflowState


def add_trace(
    state: WorkflowState,
    agent_name: str,
    action: str,
    input_data: Any = None,
    output_data: Any = None,
    error: str | None = None,
):
    state.traces.append(
        {
            "agent": agent_name,
            "action": action,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
            "error": error,
        }
    )
