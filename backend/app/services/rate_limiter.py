"""Per-user rate limiting.

Protects your OWN app - and by extension your Groq/NVIDIA/MiniMax
provider quotas - from one user firing requests fast enough to trigger
the exact 429 storms that motivated the consensus.py retry/stagger fixes
elsewhere in this codebase. Those fixes make the app recover gracefully
AFTER hitting a rate limit; this stops the limit from being hit in the
first place by capping how often any one user can call an expensive
endpoint.

In-memory, sliding-window, same single-process trade-off already made for
services/cache.py and services/automl.py's model registry - a plain dict
with a lock is enough for this deployment, and a restart just resets
everyone's window rather than losing anything meaningful.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException

_lock = threading.Lock()
# (user_id, bucket) -> deque of request timestamps within the current window
_requests: dict[tuple[str, str], deque] = {}


def check_and_record(user_id: str, bucket: str, max_requests: int, window_seconds: int) -> None:
    """Raises HTTPException(429) if this user has already made
    max_requests calls to this bucket within the last window_seconds -
    otherwise records this call and returns normally. Call this FIRST in
    an endpoint, before any expensive work happens, so a rejected request
    costs nothing beyond the check itself.
    """
    key = (user_id, bucket)
    now = time.time()
    window_start = now - window_seconds

    with _lock:
        timestamps = _requests.setdefault(key, deque())

        # Drop anything outside the current window - this is what makes
        # it a SLIDING window rather than a fixed one that resets all at
        # once (which would let a user burst right at the reset boundary).
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            retry_after = int(timestamps[0] + window_seconds - now) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many requests - limit is {max_requests} per {window_seconds}s for this "
                    f"action. Try again in about {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)


def rate_limit(bucket: str, max_requests: int, window_seconds: int):
    """FastAPI dependency factory - same DI pattern as deps.get_current_user.

    Usage: @router.post("/run", dependencies=[Depends(rate_limit("agent_run", 5, 60))])
    Must be combined with get_current_user (the endpoint's own `user=
    Depends(get_current_user)` parameter) since this needs a real user_id
    to key the limit on - it re-runs the same auth dependency internally
    rather than assuming a specific parameter name/order in the endpoint
    signature, so it works regardless of where in that signature the
    endpoint puts its own `user` parameter.
    """
    from app.deps import get_current_user
    from fastapi import Depends

    def _dependency(user=Depends(get_current_user)) -> None:
        check_and_record(user.id, bucket, max_requests, window_seconds)

    return _dependency


def reset_all() -> None:
    """Test/debug helper - clears every bucket for every user."""
    with _lock:
        _requests.clear()
