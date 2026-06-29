from typing import Protocol, Sequence

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


class PlaceRanker(Protocol):
    engine_name: str
    score_metric: str
    field_weights: dict[str, int]
    ranking_parameters: dict[str, float]

    def rank(
        self,
        query: str,
        places: Sequence[PlaceCandidate],
        filters: PlaceFilters,
        limit: int,
    ) -> list[PlaceCandidate]:
        """Rank place candidates for a normalized user query."""
