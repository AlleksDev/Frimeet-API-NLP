from typing import Protocol, Sequence

from app.modules.search.domain.filters import SearchCriteria
from app.modules.search.domain.models import SearchHit, SearchResourceType


class SearchProvider(Protocol):
    resource_type: SearchResourceType

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        offset: int,
        requester_id: str | None,
        criteria: SearchCriteria,
    ) -> Sequence[SearchHit]:
        """Return already-ranked candidates for exactly one resource type."""
