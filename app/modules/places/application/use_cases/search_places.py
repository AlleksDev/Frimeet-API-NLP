from dataclasses import dataclass, replace
import json
from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.application.ports.nearby_place_provider import NearbyPlaceProvider
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
        nearby_place_provider: NearbyPlaceProvider | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._place_repository = place_repository
        self._ranker = ranker
        self._cache = cache
        self._nearby_place_provider = nearby_place_provider

    async def execute(
        self,
        query: str,
        filters: PlaceFilters,
        limit: int = 10,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_meters: int = 5000,
    ) -> SearchPlacesResult:
        normalized_query = prepare_for_embedding(query)
        cache_key = self._cache_key(
            normalized_query,
            filters,
            limit,
            latitude,
            longitude,
            radius_meters,
        )

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        location_filter_applied = latitude is not None or longitude is not None
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be provided together")

        nearby_place_count: int | None = None
        effective_filters = filters
        if latitude is not None and longitude is not None:
            if self._nearby_place_provider is None:
                raise RuntimeError("nearby place provider is not configured")
            nearby_ids = await self._nearby_place_provider.get_nearby_place_ids(
                latitude=latitude,
                longitude=longitude,
                radius_meters=radius_meters,
            )
            nearby_place_count = len(nearby_ids)
            effective_filters = replace(
                filters,
                place_ids=tuple(sorted(nearby_ids)),
            )

        query_embedding = self._embedding_provider.embed_text(normalized_query)
        candidates = await self._place_repository.search(
            embedding=query_embedding,
            filters=effective_filters,
            limit=max(limit * 3, limit),
        )
        ranked_places = self._ranker.rank(
            query=normalized_query,
            places=candidates,
            filters=effective_filters,
            limit=limit,
        )
        result = SearchPlacesResult(
            query=query,
            normalized_query=normalized_query,
            places=ranked_places,
            metrics=self._build_metrics(
                candidates,
                ranked_places,
                location_filter_applied=location_filter_applied,
                nearby_place_count=nearby_place_count,
                radius_meters=radius_meters if location_filter_applied else None,
            ),
        )

        if self._cache:
            self._cache.set(cache_key, result)

        return result

    @staticmethod
    def _build_metrics(
        candidates: Sequence[PlaceCandidate],
        ranked_places: list[PlaceCandidate],
        location_filter_applied: bool = False,
        nearby_place_count: int | None = None,
        radius_meters: int | None = None,
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
            location_filter_applied=location_filter_applied,
            nearby_place_count=nearby_place_count,
            radius_meters=radius_meters,
        )

    @staticmethod
    def _cache_key(
        query: str,
        filters: PlaceFilters,
        limit: int,
        latitude: float | None,
        longitude: float | None,
        radius_meters: int,
    ) -> str:
        payload = {
            "query": query,
            "filters": filters.as_metadata_filter(),
            "limit": limit,
            "latitude": latitude,
            "longitude": longitude,
            "radius_meters": radius_meters,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)
