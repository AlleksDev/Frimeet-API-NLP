from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PostSourceRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    content_hash: str
    is_active: bool
    author_type: str | None = None
    author_id: str | None = None
    published_at: datetime | None = None
    source_version: int | None = None


@dataclass(frozen=True)
class PostChangeRecord:
    event_id: int
    post_id: str
    operation: str
    source_version: int
    post: PostSourceRecord | None = None


@dataclass(frozen=True)
class PostEmbeddingRecord:
    source: PostSourceRecord
    embedding: list[float]
    versioned_hash: str


@dataclass
class PostSyncResult:
    processed: int = 0
    skipped: int = 0
    upserted: int = 0
    deactivated: int = 0
    last_event_id: int = 0
    errors: list[str] = field(default_factory=list)
