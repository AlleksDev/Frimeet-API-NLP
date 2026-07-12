from collections.abc import AsyncIterator
from typing import Protocol

from app.modules.posts.domain.sync import PostChangeRecord, PostSourceRecord


class PostSyncSource(Protocol):
    def iter_snapshot(self, page_limit: int | None = None, max_pages: int | None = None) -> AsyncIterator[PostSourceRecord]: ...
    def iter_changes(self, after_id: int, page_limit: int | None = None, max_pages: int | None = None) -> AsyncIterator[PostChangeRecord]: ...
