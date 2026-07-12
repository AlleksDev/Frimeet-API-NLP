from collections.abc import AsyncIterator
from typing import Protocol

from app.modules.posts.domain.profiles import FeedInteraction


class FeedInteractionSource(Protocol):
    def iter_interactions(
        self,
        after_id: int,
        page_limit: int = 1000,
        user_id: str | None = None,
    ) -> AsyncIterator[FeedInteraction]: ...
