from dataclasses import dataclass, field
from typing import Any

from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


@dataclass(frozen=True)
class RecommendPlacesResult:
    query: str
    places: list[PlaceCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)


class RecommendPlacesUseCase:
    def __init__(self, search_use_case: SearchPlacesUseCase) -> None:
        self._search_use_case = search_use_case

    async def execute(
        self,
        query: str,
        filters: PlaceFilters,
        limit: int = 10,
    ) -> RecommendPlacesResult:
        search_result = await self._search_use_case.execute(
            query=query,
            filters=filters,
            limit=limit,
        )
        return RecommendPlacesResult(
            query=search_result.query,
            places=search_result.places,
            metadata={
                "strategy": "semantic_search_plus_ranking",
                "used_llm": False,
            },
        )
