import asyncio
import math

from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.domain.models import (
    ALL_SEARCH_RESOURCE_TYPES,
    SearchAllResult,
    SearchHit,
    SearchResourceType,
    SearchSectionPagination,
)
from app.shared.logging.config import get_logger
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding

logger = get_logger(__name__)


class SearchAllUseCase:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        providers: list[SearchProvider],
    ) -> None:
        self._embedding_provider = embedding_provider
        self._providers = {provider.resource_type: provider for provider in providers}

    async def execute(
        self,
        query: str,
        resource_types: tuple[SearchResourceType, ...] = ALL_SEARCH_RESOURCE_TYPES,
        per_type_limit: int = 5,
        top_limit: int = 10,
        requester_id: str | None = None,
        offsets: dict[SearchResourceType, int] | None = None,
    ) -> SearchAllResult:
        normalized_query = prepare_for_embedding(query)
        embedding = self._embedding_provider.embed_text(normalized_query)
        effective_offsets = offsets or {}
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
                    limit=per_type_limit + 1,
                    offset=effective_offsets.get(provider.resource_type, 0),
                    requester_id=requester_id,
                )
                for provider in providers
            ],
            return_exceptions=True,
        )

        sections: dict[SearchResourceType, list[SearchHit]] = {
            resource_type: [] for resource_type in resource_types
        }
        pagination: dict[SearchResourceType, SearchSectionPagination] = {}
        failures: dict[SearchResourceType, str] = {}
        for provider, outcome in zip(providers, outcomes):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Search provider failed resource=%s error=%s",
                    provider.resource_type,
                    type(outcome).__name__,
                )
                failures[provider.resource_type] = type(outcome).__name__
                pagination[provider.resource_type] = SearchSectionPagination(
                    page_size=per_type_limit,
                    returned_count=0,
                    has_more=False,
                )
                continue
            ranked = sorted(list(outcome), key=lambda hit: hit.score, reverse=True)
            page_hits = ranked[:per_type_limit]
            has_more = len(ranked) > per_type_limit
            current_offset = effective_offsets.get(provider.resource_type, 0)
            sections[provider.resource_type] = page_hits
            pagination[provider.resource_type] = SearchSectionPagination(
                page_size=per_type_limit,
                returned_count=len(page_hits),
                has_more=has_more,
                next_offset=(current_offset + len(page_hits) if has_more else None),
            )

        top_results = _diversified_top_results(sections, top_limit)
        return SearchAllResult(
            query=query,
            normalized_query=normalized_query,
            top_results=top_results,
            sections=sections,
            pagination=pagination,
            failed_resources=failures,
        )


def _diversified_top_results(
    sections: dict[SearchResourceType, list[SearchHit]],
    limit: int,
) -> list[SearchHit]:
    if limit <= 0:
        return []
    candidates = sorted(
        (hit for hits in sections.values() for hit in hits),
        key=lambda hit: hit.score,
        reverse=True,
    )
    max_per_type = max(2, math.ceil(limit / 3))
    counts: dict[SearchResourceType, int] = {}
    selected: list[SearchHit] = []
    for hit in candidates:
        if counts.get(hit.resource_type, 0) >= max_per_type:
            continue
        selected.append(hit)
        counts[hit.resource_type] = counts.get(hit.resource_type, 0) + 1
        if len(selected) >= limit:
            break
    return selected
