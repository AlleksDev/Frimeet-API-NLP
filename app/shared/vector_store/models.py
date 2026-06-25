from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None


@dataclass(frozen=True)
class VectorUpsertRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]
    content_hash: str
    is_active: bool
