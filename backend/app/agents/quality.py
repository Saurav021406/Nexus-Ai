"""Backward-compatible fused Reviewer + Security check.

The new intent-aware path (agents/manager_v2.py) now calls review_check()
and security_check() directly as two separate agents/steps (Phase 4
roadmap: "Separate Reviewer & Security agents") - see agents/reviewer.py
and agents/security.py.

This wrapper only exists so the OLD domain-routing path
(agents/manager.py -> /domain/analyze, the original "AI analysis" tab)
keeps working completely unchanged - it still expects one
quality_check(state) call returning {"review": ..., "security": ...}.
"""

from concurrent.futures import ThreadPoolExecutor

from app.agents.reviewer import review_check
from app.agents.security import security_check
from app.agents.state import WorkflowState


def quality_check(state: WorkflowState) -> dict:
    """Runs the reviewer and security checks in parallel and merges them
    into the shape callers already expect."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        review_future = pool.submit(review_check, state)
        security_future = pool.submit(security_check, state)
        return {"review": review_future.result(), "security": security_future.result()}
