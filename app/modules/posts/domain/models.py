from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PostCandidate:
    id: str
    title: str
    score: float
    city: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
