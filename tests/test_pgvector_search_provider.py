from datetime import UTC, datetime

import pytest

from app.modules.search.domain.filters import SearchCriteria
from app.modules.search.domain.models import SearchResourceType
from app.modules.search.infrastructure.pgvector_provider import (
    PgvectorHybridSearchProvider,
    PgvectorPlaceSearchProvider,
)
from app.modules.search.domain.relevance import SearchRelevancePolicy
from app.shared.vector_store.models import VectorMatch


class RecordingVectorClient:
    def __init__(self) -> None:
        self.match_places_calls: list[dict[str, object]] = []

    async def match_places(
        self,
        embedding: list[float],
        filters: dict[str, object],
        limit: int,
    ) -> list[VectorMatch]:
        self.match_places_calls.append(
            {"embedding": embedding, "filters": filters, "limit": limit}
        )
        return [
            VectorMatch(
                id="place-1",
                score=0.82,
                metadata={"name": "Cafe Central", "category": "cafe"},
                document="cafe tranquilo para trabajar",
            ),
            VectorMatch(
                id="place-2",
                score=0.75,
                metadata={"name": "Cafe Sur", "category": "cafe"},
                document="cafe para conversar",
            ),
        ][:limit]

    async def search_resource_embeddings(self, **kwargs: object) -> list[VectorMatch]:
        raise AssertionError("Places must not use the hybrid RRF search function")


@pytest.mark.asyncio
async def test_places_use_same_cosine_match_function_as_recommendations() -> None:
    vector_client = RecordingVectorClient()
    provider = PgvectorPlaceSearchProvider(vector_client)  # type: ignore[arg-type]

    hits = await provider.search(
        query="cafe tranquilo",
        embedding=[0.1, 0.2, 0.3],
        limit=1,
        offset=1,
        requester_id=None,
        criteria=SearchCriteria(),
    )

    assert vector_client.match_places_calls == [
        {
            "embedding": [0.1, 0.2, 0.3],
            "filters": {"is_active": True},
            "limit": 2,
        }
    ]
    assert hits[0].id == "place-2"
    assert hits[0].resource_type == SearchResourceType.PLACES
    assert hits[0].score == 0.75
    assert hits[0].semantic_score == 0.75
    assert hits[0].lexical_score is None


class RecordingHybridVectorClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def search_resource_embeddings(self, **kwargs: object) -> list[VectorMatch]:
        self.calls.append(dict(kwargs))
        return [
            VectorMatch(
                id="rejected",
                score=0.9,
                semantic_score=0.39,
                lexical_score=0.09,
                metadata={
                    "start_time": "2026-07-12T11:30:00Z",
                    "duration_minutes": 120,
                },
            ),
            VectorMatch(
                id="lexical",
                score=0.8,
                semantic_score=0.20,
                lexical_score=0.10,
                metadata={
                    "start_time": "2026-07-12T11:30:00Z",
                    "duration_minutes": 120,
                },
            ),
        ]


@pytest.mark.asyncio
async def test_hybrid_provider_pushes_event_and_threshold_filters_to_sql() -> None:
    vector_client = RecordingHybridVectorClient()
    policy = SearchRelevancePolicy.uniform(semantic_min=0.40, lexical_min=0.10)
    provider = PgvectorHybridSearchProvider(  # type: ignore[arg-type]
        SearchResourceType.EVENTS,
        vector_client,
        policy,
    )
    active_at = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)

    hits = await provider.search(
        query="evento",
        embedding=[0.1, 0.2, 0.3],
        limit=5,
        offset=0,
        requester_id=None,
        criteria=SearchCriteria(event_active_at=active_at),
    )

    assert [hit.id for hit in hits] == ["lexical"]
    filters = vector_client.calls[0]["filters"]
    assert filters == {
        "is_active": True,
        "min_semantic_score": 0.40,
        "min_lexical_score": 0.10,
        "event_active_at": active_at.isoformat(),
    }
