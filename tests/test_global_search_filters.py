from datetime import UTC, datetime

import pytest

from app.modules.search.application.use_cases.search_all import SearchAllUseCase
from app.modules.search.domain.filters import (
    ClubAttendanceMode,
    GlobalSearchFilters,
    LocationSearchMode,
    SearchCriteria,
    SearchLocation,
    filter_and_rank_hits,
)
from app.modules.search.domain.models import SearchHit, SearchResourceType
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider


def _hit(
    hit_id: str,
    resource_type: SearchResourceType,
    score: float,
    **metadata: object,
) -> SearchHit:
    return SearchHit(
        id=hit_id,
        resource_type=resource_type,
        title=hit_id,
        score=score,
        metadata=dict(metadata),
    )


def test_nearby_places_are_prioritized_without_changing_semantic_score() -> None:
    criteria = SearchCriteria(
        location=SearchLocation(16.75, -93.11, mode=LocationSearchMode.PRIORITIZE),
        nearby_place_ids=frozenset({"nearby"}),
    )
    ranked = filter_and_rank_hits(
        SearchResourceType.PLACES,
        [
            _hit("far", SearchResourceType.PLACES, 0.80),
            _hit("nearby", SearchResourceType.PLACES, 0.72),
        ],
        criteria,
    )

    assert [hit.id for hit in ranked] == ["nearby", "far"]
    assert ranked[0].score == 0.72
    assert ranked[0].is_nearby is True
    assert ranked[0].proximity_boost == 0.12


def test_strict_radius_excludes_non_nearby_location_resources() -> None:
    criteria = SearchCriteria(
        location=SearchLocation(16.75, -93.11, mode=LocationSearchMode.STRICT),
        nearby_place_ids=frozenset({"place-near"}),
    )
    ranked = filter_and_rank_hits(
        SearchResourceType.CLUBS,
        [
            _hit("club-near", SearchResourceType.CLUBS, 0.60, place_id="place-near"),
            _hit("club-far", SearchResourceType.CLUBS, 0.90, place_id="place-far"),
        ],
        criteria,
    )

    assert [hit.id for hit in ranked] == ["club-near"]


def test_resource_specific_filters_only_apply_to_supported_resources() -> None:
    criteria = SearchCriteria(
        filters=GlobalSearchFilters(
            price_ranges=("$$",),
            club_mode=ClubAttendanceMode.ONLINE,
            published_from=datetime(2026, 7, 1, tzinfo=UTC),
            published_to=datetime(2026, 7, 31, tzinfo=UTC),
        )
    )

    places = filter_and_rank_hits(
        SearchResourceType.PLACES,
        [
            _hit("cheap", SearchResourceType.PLACES, 0.9, price_range="$"),
            _hit("accepted", SearchResourceType.PLACES, 0.8, price_range="$$"),
        ],
        criteria,
    )
    clubs = filter_and_rank_hits(
        SearchResourceType.CLUBS,
        [
            _hit("physical", SearchResourceType.CLUBS, 0.9, is_online=False),
            _hit("online", SearchResourceType.CLUBS, 0.8, is_online=True),
        ],
        criteria,
    )
    posts = filter_and_rank_hits(
        SearchResourceType.POSTS,
        [
            _hit(
                "old",
                SearchResourceType.POSTS,
                0.9,
                published_at="2026-06-01T00:00:00Z",
            ),
            _hit(
                "current",
                SearchResourceType.POSTS,
                0.8,
                published_at="2026-07-04T00:00:00Z",
            ),
        ],
        criteria,
    )

    assert [hit.id for hit in places] == ["accepted"]
    assert [hit.id for hit in clubs] == ["online"]
    assert [hit.id for hit in posts] == ["current"]


class RecordingNearbyProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, int]] = []

    async def get_nearby_place_ids(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> set[str]:
        self.calls.append((latitude, longitude, radius_meters))
        return {"nearby"}


class CriteriaRecordingProvider:
    resource_type = SearchResourceType.PLACES

    def __init__(self) -> None:
        self.criteria: SearchCriteria | None = None

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        offset: int,
        requester_id: str | None,
        criteria: SearchCriteria,
    ) -> list[SearchHit]:
        self.criteria = criteria
        return []


@pytest.mark.asyncio
async def test_search_resolves_nearby_ids_once_before_fanning_out() -> None:
    nearby_provider = RecordingNearbyProvider()
    search_provider = CriteriaRecordingProvider()
    use_case = SearchAllUseCase(
        embedding_provider=MockEmbeddingProvider(dimension=8),
        providers=[search_provider],
        nearby_place_provider=nearby_provider,
    )

    await use_case.execute(
        query="cafe",
        resource_types=(SearchResourceType.PLACES,),
        criteria=SearchCriteria(
            location=SearchLocation(16.75, -93.11, radius_meters=3000)
        ),
    )

    assert nearby_provider.calls == [(16.75, -93.11, 3000)]
    assert search_provider.criteria is not None
    assert search_provider.criteria.nearby_place_ids == frozenset({"nearby"})
