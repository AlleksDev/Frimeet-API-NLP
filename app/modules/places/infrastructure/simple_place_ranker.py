from typing import Sequence

from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


class SimplePlaceRanker(PlaceRanker):
    def rank(
        self,
        query: str,
        places: Sequence[PlaceCandidate],
        filters: PlaceFilters,
        limit: int,
    ) -> list[PlaceCandidate]:
        del query
        ranked = sorted(
            places,
            key=lambda place: (
                place.score,
                _city_boost(place, filters.city),
                _category_boost(place, filters.category),
            ),
            reverse=True,
        )
        return ranked[:limit]


def _city_boost(place: PlaceCandidate, city: str | None) -> float:
    if not city or not place.city:
        return 0.0
    return 0.05 if place.city.casefold() == city.casefold() else 0.0


def _category_boost(place: PlaceCandidate, category: str | None) -> float:
    if not category or not place.category:
        return 0.0
    return 0.03 if place.category.casefold() == category.casefold() else 0.0
