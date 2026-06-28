from dataclasses import dataclass, replace
import json
from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.application.ports.nearby_place_provider import NearbyPlaceProvider
from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_metrics import SearchEngineMetrics
from app.modules.places.domain.search_text import place_tokens, tokenize
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
        relevance_threshold: float = 3.0,
        no_match_threshold: float = 0.0,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._place_repository = place_repository
        self._ranker = ranker
        self._cache = cache
        self._nearby_place_provider = nearby_place_provider
        self._relevance_threshold = relevance_threshold
        self._no_match_threshold = no_match_threshold

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
                normalized_query,
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

    def _build_metrics(
        self,
        query: str,
        candidates: Sequence[PlaceCandidate],
        ranked_places: list[PlaceCandidate],
        location_filter_applied: bool = False,
        nearby_place_count: int | None = None,
        radius_meters: int | None = None,
    ) -> SearchEngineMetrics:
        scores = [place.score for place in ranked_places]
        max_score = max(scores, default=0.0)
        query_terms = set(tokenize(query))
        candidate_terms = {
            term
            for candidate in candidates
            for term in place_tokens(candidate)
        }
        matched_query_terms = query_terms & candidate_terms
        return SearchEngineMetrics(
            engine=self._ranker.engine_name,
            candidate_retrieval=self._place_repository.source_name,
            score_metric=self._ranker.score_metric,
            field_weights=self._ranker.field_weights,
            ranking_parameters=self._ranker.ranking_parameters,
            relevance_threshold=self._relevance_threshold,
            match_quality=_match_quality(
                max_score,
                self._relevance_threshold,
                self._no_match_threshold,
            ),
            query_token_count=len(query_terms),
            matched_query_token_count=len(matched_query_terms),
            query_coverage=(
                len(matched_query_terms) / len(query_terms)
                if query_terms
                else 0.0
            ),
            scope="current_query",
            ground_truth_available=False,
            candidate_count=len(candidates),
            returned_count=len(ranked_places),
            nonzero_score_count=sum(score > 0 for score in scores),
            min_score=min(scores, default=0.0),
            max_score=max_score,
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


def _match_quality(
    max_score: float,
    relevance_threshold: float,
    no_match_threshold: float = 0.0,
) -> str:
    if max_score <= no_match_threshold:
        return "no_match"
    if max_score < relevance_threshold:
        return "low_confidence"
    return "confident"
