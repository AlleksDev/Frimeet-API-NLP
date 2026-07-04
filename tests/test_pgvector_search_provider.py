import pytest

from app.modules.search.domain.filters import SearchCriteria
from app.modules.search.domain.models import SearchResourceType
from app.modules.search.infrastructure.pgvector_provider import (
    PgvectorPlaceSearchProvider,
)
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
