from typing import Protocol, Sequence

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


class PlaceVectorRepository(Protocol):
    source_name: str

    async def search(
        self,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        """Search place vectors using an already generated query embedding."""
