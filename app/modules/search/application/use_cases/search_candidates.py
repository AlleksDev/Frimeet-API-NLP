import asyncio
import logging
from datetime import datetime

from app.modules.search.application.ports.nearby_place_provider import (
    NearbyPlaceProvider,
)
from app.modules.search.application.ports.query_embedding_provider import (
    QueryEmbeddingProvider,
)
from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.domain.filters import SearchCriteria
from app.modules.search.domain.models import (
    SearchCandidate,
    SearchCandidatesResult,
    SearchResourceType,
    SearchSectionPagination,
)
from app.modules.search.domain.query import normalize_search_query

logger = logging.getLogger(__name__)


class SearchCandidatesUseCase:
    def __init__(
        self,
        embedding_provider: QueryEmbeddingProvider,
        providers: list[SearchProvider],
        policy_version: str,
        nearby_place_provider: NearbyPlaceProvider | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._providers = {provider.resource_type: provider for provider in providers}
        self._nearby_place_provider = nearby_place_provider
        self._policy_version = policy_version

    async def execute(
        self,
        query: str,
        resource_types: tuple[SearchResourceType, ...],
        candidate_limit_per_type: int,
        as_of: datetime,
        offsets: dict[SearchResourceType, int] | None = None,
        criteria: SearchCriteria | None = None,
        cursor_context: str = "",
    ) -> SearchCandidatesResult:
        normalized_query = normalize_search_query(query)
        embedding = self._embedding_provider.embed_text(normalized_query)
        effective_offsets = offsets or {}
        effective_criteria = (criteria or SearchCriteria()).with_event_active_at(as_of)
        if effective_criteria.location is not None:
            if self._nearby_place_provider is None:
                raise RuntimeError("nearby place provider is not configured")
            nearby_ids = await self._nearby_place_provider.get_nearby_place_ids(
                latitude=effective_criteria.location.latitude,
                longitude=effective_criteria.location.longitude,
                radius_meters=effective_criteria.location.radius_meters,
            )
            effective_criteria = effective_criteria.with_nearby_place_ids(nearby_ids)

        providers = [
            self._providers[resource_type]
            for resource_type in resource_types
            if resource_type in self._providers
        ]
        outcomes = await asyncio.gather(
            *[
                provider.search(
                    query=normalized_query,
                    embedding=embedding,
                    limit=candidate_limit_per_type + 1,
                    offset=effective_offsets.get(provider.resource_type, 0),
                    requester_id=None,
                    criteria=effective_criteria,
                )
                for provider in providers
            ],
            return_exceptions=True,
        )

        sections: dict[SearchResourceType, list[SearchCandidate]] = {
            resource_type: [] for resource_type in resource_types
        }
        pagination: dict[SearchResourceType, SearchSectionPagination] = {}
        failures: dict[SearchResourceType, str] = {}
        for provider, outcome in zip(providers, outcomes):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Internal search provider failed resource=%s error=%s",
                    provider.resource_type,
                    type(outcome).__name__,
                )
                failures[provider.resource_type] = type(outcome).__name__
                pagination[provider.resource_type] = SearchSectionPagination(
                    page_size=candidate_limit_per_type,
                    returned_count=0,
                    has_more=False,
                )
                continue
            ranked = list(outcome)
            page_hits = ranked[:candidate_limit_per_type]
            has_more = len(ranked) > candidate_limit_per_type
            current_offset = effective_offsets.get(provider.resource_type, 0)
            sections[provider.resource_type] = [
                SearchCandidate(
                    id=hit.id,
                    resource_type=hit.resource_type,
                    score=hit.score,
                    semantic_score=hit.semantic_score,
                    lexical_score=hit.lexical_score,
                )
                for hit in page_hits
            ]
            pagination[provider.resource_type] = SearchSectionPagination(
                page_size=candidate_limit_per_type,
                returned_count=len(page_hits),
                has_more=has_more,
                next_offset=(current_offset + len(page_hits) if has_more else None),
            )

        for resource_type in resource_types:
            pagination.setdefault(
                resource_type,
                SearchSectionPagination(
                    page_size=candidate_limit_per_type,
                    returned_count=0,
                    has_more=False,
                ),
            )
        return SearchCandidatesResult(
            query=query,
            normalized_query=normalized_query,
            sections=sections,
            pagination=pagination,
            failed_resources=failures,
            cursor_context=cursor_context,
            policy_version=self._policy_version,
        )
