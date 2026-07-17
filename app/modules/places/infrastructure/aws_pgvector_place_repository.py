from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorMatch


class AwsPgvectorPlaceRepository(PlaceVectorRepository):
    source_name = "pgvector"

    def __init__(
        self,
        vector_client: AwsPgvectorClient,
        match_function: str = "match_places",
        hybrid_function: str | None = None,
    ) -> None:
        self._vector_client = vector_client
        self._match_function = match_function
        self._hybrid_function = hybrid_function

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
            function_name=self._match_function,
        )
        return [_match_to_candidate(match) for match in matches]

    async def search_hybrid(
        self,
        query_text: str,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        """Use the versioned SQL hybrid contract when it is configured."""

        if not self._hybrid_function:
            return await self.search(embedding=embedding, filters=filters, limit=limit)
        metadata_filter = filters.as_metadata_filter()
        metadata_filter["is_active"] = True
        matches = await self._vector_client.search_places_hybrid(
            query_text=query_text,
            embedding=embedding,
            filters=metadata_filter,
            limit=limit,
            function_name=self._hybrid_function,
        )
        return [_match_to_candidate(match) for match in matches]


def _match_to_candidate(match: VectorMatch) -> PlaceCandidate:
    metadata = dict(match.metadata)
    if match.semantic_score is not None:
        metadata["semantic_score"] = match.semantic_score
    if match.lexical_score is not None:
        metadata["lexical_score"] = match.lexical_score
    return PlaceCandidate(
        id=match.id,
        name=str(metadata.get("name") or match.id),
        score=match.score,
        category=metadata.get("category"),
        city=metadata.get("city"),
        state=metadata.get("state"),
        price_range=metadata.get("price_range"),
        metadata=metadata,
        document=match.document,
    )
