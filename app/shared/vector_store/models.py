from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None


@dataclass(frozen=True)
class VectorUpsertRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]
    content_hash: str
    is_active: bool
    author_type: str | None = None
    author_id: str | None = None
    published_at: datetime | None = None
    source_version: int | None = None
