from dataclasses import dataclass
import json

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.cache.memory import SimpleTTLCache
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


@dataclass(frozen=True)
class SearchPlacesResult:
    query: str
    normalized_query: str
    places: list[PlaceCandidate]


class SearchPlacesUseCase:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        place_repository: PlaceVectorRepository,
        ranker: PlaceRanker,
        cache: SimpleTTLCache | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._place_repository = place_repository
        self._ranker = ranker
        self._cache = cache

    async def execute(
        self,
        query: str,
        filters: PlaceFilters,
        limit: int = 10,
    ) -> SearchPlacesResult:
        normalized_query = prepare_for_embedding(query)
        cache_key = self._cache_key(normalized_query, filters, limit)

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        query_embedding = self._embedding_provider.embed_text(normalized_query)
        candidates = await self._place_repository.search(
            embedding=query_embedding,
            filters=filters,
            limit=max(limit * 3, limit),
        )
        ranked_places = self._ranker.rank(
            query=normalized_query,
            places=candidates,
            filters=filters,
            limit=limit,
        )
        result = SearchPlacesResult(
            query=query,
            normalized_query=normalized_query,
            places=ranked_places,
        )

        if self._cache:
            self._cache.set(cache_key, result)

        return result

    @staticmethod
    def _cache_key(query: str, filters: PlaceFilters, limit: int) -> str:
        payload = {
            "query": query,
            "filters": filters.as_metadata_filter(),
            "limit": limit,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)
