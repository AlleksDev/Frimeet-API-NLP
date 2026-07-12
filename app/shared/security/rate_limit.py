from collections import deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status

from app.shared.config.settings import get_settings


_requests: dict[str, deque[float]] = {}
_lock = Lock()
_MAX_BUCKETS = 10_000


async def rate_limit_placeholder(request: Request) -> None:
    """Apply a small in-process fixed-window guard at the HTTP boundary.

    The production gateway may impose an additional distributed limit. This guard is
    intentionally local to each worker so a missing gateway rule cannot leave the
    service completely unbounded.
    """
    settings = get_settings()
    window_seconds = settings.rate_limit_window_seconds
    limit = (
        settings.internal_rate_limit_requests_per_window
        if request.url.path.startswith("/internal/")
        else settings.rate_limit_requests_per_window
    )
    client_host = request.client.host if request.client else "unknown"
    key = f"{client_host}:{request.url.path}"
    now = monotonic()
    cutoff = now - window_seconds

    with _lock:
        if key not in _requests and len(_requests) >= _MAX_BUCKETS:
            stale_keys = [
                bucket_key
                for bucket_key, bucket in _requests.items()
                if not bucket or bucket[-1] <= cutoff
            ]
            for stale_key in stale_keys:
                _requests.pop(stale_key, None)
            if len(_requests) >= _MAX_BUCKETS:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit capacity exceeded",
                    headers={"Retry-After": str(window_seconds)},
                )
        timestamps = _requests.setdefault(key, deque())
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= limit:
            retry_after = max(1, int(window_seconds - (now - timestamps[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)


def reset_rate_limit_state() -> None:
    """Clear process-local buckets for deterministic tests."""
    with _lock:
        _requests.clear()
