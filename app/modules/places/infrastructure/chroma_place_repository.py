from typing import Any, Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.chroma.vector_store import ChromaVectorStore


class ChromaPlaceVectorRepository(PlaceVectorRepository):
    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    async def search(
        self,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        raw_results = await self._vector_store.query(
            embedding=embedding,
            metadata_filter=filters.as_metadata_filter(),
            limit=limit,
        )
        return self._to_candidates(raw_results)

    def _to_candidates(self, raw_results: dict[str, Any]) -> list[PlaceCandidate]:
        ids = raw_results.get("ids", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        candidates: list[PlaceCandidate] = []
        for index, place_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            candidates.append(
                PlaceCandidate(
                    id=str(place_id),
                    name=str(metadata.get("name", place_id)),
                    score=score,
                    category=metadata.get("category"),
                    city=metadata.get("city"),
                    state=metadata.get("state"),
                    price_range=metadata.get("price_range"),
                    metadata=dict(metadata),
                )
            )
        return candidates
