from typing import Protocol, Sequence

from app.modules.search.domain.models import SearchHit, SearchResourceType


class SearchProvider(Protocol):
    resource_type: SearchResourceType

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        requester_id: str | None,
    ) -> Sequence[SearchHit]:
        """Return already-ranked candidates for exactly one resource type."""
