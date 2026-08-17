"""Lightweight event shape used to report live workflow progress
(Section 30: "Build/update the Nexus frontend to show live orchestration").

This is deliberately separate from agents/tracing.py's add_trace(), which
is the permanent record written into WorkflowState.traces. Events here are
transient - they exist only to be streamed to a connected frontend while a
workflow is running. A workflow that isn't being watched live still works
identically; nothing here is required for correctness.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

EventEmitter = Callable[[str, str, str, dict | None], Awaitable[None]]


def make_event(event_type: str, agent: str, message: str, data: dict[str, Any] | None = None) -> dict:
    return {
        "type": event_type,
        "agent": agent,
        "message": message,
        "data": data,
        "timestamp": time.time(),
    }


async def _noop_emit(event_type: str, agent: str, message: str, data: dict | None = None) -> None:
    return None


NOOP_EMIT: EventEmitter = _noop_emit
