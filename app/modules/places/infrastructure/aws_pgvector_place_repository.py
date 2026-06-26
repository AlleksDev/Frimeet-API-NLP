from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorMatch


class AwsPgvectorPlaceRepository(PlaceVectorRepository):
    def __init__(self, vector_client: AwsPgvectorClient) -> None:
        self._vector_client = vector_client

    async def search(
        self,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        metadata_filter = filters.as_metadata_filter()
        metadata_filter["is_active"] = True
        matches = await self._vector_client.match_places(
            embedding=embedding,
            filters=metadata_filter,
            limit=limit,
        )
        return [_match_to_candidate(match) for match in matches]


def _match_to_candidate(match: VectorMatch) -> PlaceCandidate:
    metadata = match.metadata
    return PlaceCandidate(
        id=match.id,
        name=str(metadata.get("name") or match.id),
        score=match.score,
        category=metadata.get("category"),
        city=metadata.get("city"),
        state=metadata.get("state"),
        price_range=metadata.get("price_range"),
        metadata=metadata,
    )
