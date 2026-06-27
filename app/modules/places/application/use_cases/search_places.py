from dataclasses import dataclass
import json
from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_metrics import SearchEngineMetrics
from app.shared.cache.memory import SimpleTTLCache
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


@dataclass(frozen=True)
class SearchPlacesResult:
    query: str
    normalized_query: str
    places: list[PlaceCandidate]
    metrics: SearchEngineMetrics


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
            metrics=self._build_metrics(candidates, ranked_places),
        )

        if self._cache:
            self._cache.set(cache_key, result)

        return result

    @staticmethod
    def _build_metrics(
        candidates: Sequence[PlaceCandidate],
        ranked_places: list[PlaceCandidate],
    ) -> SearchEngineMetrics:
        scores = [place.score for place in ranked_places]
        return SearchEngineMetrics(
            engine="tfidf",
            candidate_retrieval="embeddings",
            score_metric="cosine_similarity",
            field_weights={
                "tags": 6,
                "category": 2,
                "other_text": 1,
            },
            candidate_count=len(candidates),
            returned_count=len(ranked_places),
            nonzero_score_count=sum(score > 0 for score in scores),
            min_score=min(scores, default=0.0),
            max_score=max(scores, default=0.0),
            mean_score=sum(scores) / len(scores) if scores else 0.0,
        )

    @staticmethod
    def _cache_key(query: str, filters: PlaceFilters, limit: int) -> str:
        payload = {
            "query": query,
            "filters": filters.as_metadata_filter(),
            "limit": limit,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)
