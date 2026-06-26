from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class SimpleTTLCache:
    def __init__(self, default_ttl_seconds: int = 60) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._values: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        if entry.expires_at <= monotonic():
            self._values.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        self._values[key] = _CacheEntry(
            value=value,
            expires_at=monotonic() + ttl,
        )

    def clear(self) -> None:
        self._values.clear()
