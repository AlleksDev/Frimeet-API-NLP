from typing import Protocol, Sequence

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


class PlaceRanker(Protocol):
    def rank(
        self,
        places: Sequence[PlaceCandidate],
        filters: PlaceFilters,
        limit: int,
    ) -> list[PlaceCandidate]:
        """Rank candidates returned by the vector store."""
