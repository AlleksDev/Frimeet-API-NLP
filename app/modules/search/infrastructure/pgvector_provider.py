from typing import Sequence

from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.domain.filters import (
    LocationSearchMode,
    SearchCriteria,
    filter_and_rank_hits,
)
from app.modules.search.domain.models import SearchHit, SearchResourceType
from app.modules.search.domain.relevance import SearchRelevancePolicy
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorMatch

_PRIVATE_METADATA_KEYS = {"authorized_user_ids", "creator_id"}


class PgvectorPlaceSearchProvider(SearchProvider):
    """Combine lexical and FastText evidence for global place search."""

    resource_type = SearchResourceType.PLACES

    def __init__(
        self,
        vector_client: AwsPgvectorClient,
        relevance_policy: SearchRelevancePolicy | None = None,
    ) -> None:
        self._vector_client = vector_client
        self._relevance_policy = relevance_policy or SearchRelevancePolicy.uniform()

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        offset: int,
        requester_id: str | None,
        criteria: SearchCriteria,
    ) -> Sequence[SearchHit]:
        del requester_id
        if (
            criteria.location is not None
            and criteria.location.mode == LocationSearchMode.STRICT
            and not criteria.nearby_place_ids
        ):
            return []
        metadata_filters = _place_metadata_filters(criteria)
        fetch_limit = _candidate_limit(offset, limit, criteria, self.resource_type)
        if (
            criteria.location is not None
            and criteria.location.mode == LocationSearchMode.STRICT
        ):
            # The legacy matcher filters Places by external_id. The generic
            # hybrid contract uses metadata.place_id, which belongs to related
            # resources such as clubs and events, so strict radius keeps the
            # existing SQL path to avoid widening the requested area.
            matches = await self._vector_client.match_places(
                embedding=embedding,
                filters=metadata_filters,
                limit=fetch_limit,
            )
        else:
            threshold = self._relevance_policy.threshold_for(self.resource_type)
            metadata_filters["min_semantic_score"] = threshold.semantic_min
            metadata_filters["min_lexical_score"] = threshold.lexical_min
            matches = await self._vector_client.search_resource_embeddings(
                resource_type=self.resource_type.value,
                query_text=query,
                embedding=embedding,
                filters=metadata_filters,
                limit=fetch_limit,
            )
        hits = [
            _to_search_hit(self.resource_type, match)
            for match in matches
        ]
        relevant_hits = [hit for hit in hits if self._relevance_policy.accepts(hit)]
        ranked = filter_and_rank_hits(self.resource_type, relevant_hits, criteria)
        return ranked[offset : offset + limit]


class PgvectorHybridSearchProvider(SearchProvider):
    def __init__(
        self,
        resource_type: SearchResourceType,
        vector_client: AwsPgvectorClient,
        relevance_policy: SearchRelevancePolicy | None = None,
    ) -> None:
        self.resource_type = resource_type
        self._vector_client = vector_client
        self._relevance_policy = relevance_policy or SearchRelevancePolicy.uniform()

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        offset: int,
        requester_id: str | None,
        criteria: SearchCriteria,
    ) -> Sequence[SearchHit]:
        if (
            criteria.location is not None
            and criteria.location.mode == LocationSearchMode.STRICT
            and self.resource_type in {SearchResourceType.CLUBS, SearchResourceType.EVENTS}
            and not criteria.nearby_place_ids
        ):
            return []
        filters = {"is_active": True}
        threshold = self._relevance_policy.threshold_for(self.resource_type)
        filters["min_semantic_score"] = threshold.semantic_min
        filters["min_lexical_score"] = threshold.lexical_min
        if (
            self.resource_type == SearchResourceType.EVENTS
            and criteria.event_active_at is not None
        ):
            filters["event_active_at"] = criteria.event_active_at.isoformat()
        if requester_id:
            filters["requester_id"] = requester_id
        filters.update(_hybrid_metadata_filters(criteria, self.resource_type))
        fetch_limit = _candidate_limit(offset, limit, criteria, self.resource_type)
        matches = await self._vector_client.search_resource_embeddings(
            resource_type=self.resource_type.value,
            query_text=query,
            embedding=embedding,
            filters=filters,
            limit=fetch_limit,
        )
        hits = [
            _to_search_hit(self.resource_type, match)
            for match in matches
        ]
        relevant_hits = [hit for hit in hits if self._relevance_policy.accepts(hit)]
        ranked = filter_and_rank_hits(self.resource_type, relevant_hits, criteria)
        return ranked[offset : offset + limit]


def _candidate_limit(
    offset: int,
    limit: int,
    criteria: SearchCriteria,
    resource_type: SearchResourceType,
) -> int:
    requested = offset + limit
    if not criteria.requires_extended_candidates(resource_type):
        return requested
    return min(max(requested * 5, 100), 1000)


def _place_metadata_filters(criteria: SearchCriteria) -> dict[str, object]:
    filters: dict[str, object] = {"is_active": True}
    if criteria.filters.city:
        filters["city"] = criteria.filters.city
    if criteria.filters.state:
        filters["state"] = criteria.filters.state
    if criteria.filters.categories:
        filters["categories"] = list(criteria.filters.categories)
    if criteria.filters.price_ranges:
        filters["price_ranges"] = list(criteria.filters.price_ranges)
    if criteria.filters.tags:
        filters["tags"] = list(criteria.filters.tags)
    if (
        criteria.location is not None
        and criteria.location.mode == LocationSearchMode.STRICT
    ):
        filters["place_ids"] = sorted(criteria.nearby_place_ids)
    return filters


def _hybrid_metadata_filters(
    criteria: SearchCriteria,
    resource_type: SearchResourceType,
) -> dict[str, object]:
    source = criteria.filters
    filters: dict[str, object] = {}
    if resource_type == SearchResourceType.POSTS:
        if source.city:
            filters["city"] = source.city
        if source.state:
            filters["state"] = source.state
        if source.tags:
            filters["tags"] = list(source.tags)
        if source.published_from:
            filters["published_from"] = source.published_from.isoformat()
        if source.published_to:
            filters["published_to"] = source.published_to.isoformat()
    elif resource_type == SearchResourceType.CLUBS:
        if source.categories:
            filters["categories"] = list(source.categories)
        if source.club_mode is not None:
            filters["is_online"] = source.club_mode.value == "online"
    elif resource_type == SearchResourceType.EVENTS:
        if source.tags:
            filters["tags"] = list(source.tags)
        if source.event_from:
            filters["event_from"] = source.event_from.isoformat()
        if source.event_to:
            filters["event_to"] = source.event_to.isoformat()
    elif resource_type == SearchResourceType.USERS and source.user_roles:
        filters["user_roles"] = list(source.user_roles)

    if (
        criteria.location is not None
        and criteria.location.mode == LocationSearchMode.STRICT
        and resource_type in {SearchResourceType.CLUBS, SearchResourceType.EVENTS}
    ):
        filters["place_ids"] = sorted(criteria.nearby_place_ids)
    return filters


def _to_search_hit(resource_type: SearchResourceType, match: VectorMatch) -> SearchHit:
    metadata = {
        key: value
        for key, value in match.metadata.items()
        if key not in _PRIVATE_METADATA_KEYS
    }
    title = str(
        metadata.get("search_title")
        or metadata.get("name")
        or metadata.get("title")
        or metadata.get("full_name")
        or metadata.get("username")
        or match.id
    )
    subtitle = _subtitle(resource_type, metadata)
    semantic_score = match.semantic_score
    if resource_type == SearchResourceType.PLACES and semantic_score is None:
        semantic_score = match.score
    return SearchHit(
        id=match.id,
        resource_type=resource_type,
        title=title,
        subtitle=subtitle,
        score=match.score,
        semantic_score=semantic_score,
        lexical_score=match.lexical_score,
        metadata=metadata,
    )


def _subtitle(resource_type: SearchResourceType, metadata: dict[str, object]) -> str | None:
    if resource_type == SearchResourceType.USERS:
        username = metadata.get("username")
        return f"@{username}" if username else None
    for key in ("category", "description", "short_description", "city"):
        value = metadata.get(key)
        if value:
            return str(value)[:160]
    return None
