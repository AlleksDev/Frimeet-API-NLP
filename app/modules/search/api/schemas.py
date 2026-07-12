from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.search.api.cursor import MAX_SEARCH_OFFSET, encode_search_cursor
from app.modules.search.domain.filters import (
    ClubAttendanceMode,
    GlobalSearchFilters,
    LocationSearchMode,
    SearchCriteria,
    SearchLocation,
)
from app.modules.search.domain.models import SearchAllResult, SearchHit, SearchResourceType
from app.modules.search.domain.query import normalize_search_query


class SearchLocationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    radius: int = Field(default=5000, ge=1, le=50_000)
    mode: LocationSearchMode = LocationSearchMode.PRIORITIZE

    def to_domain(self) -> SearchLocation:
        return SearchLocation(
            latitude=self.lat,
            longitude=self.lng,
            radius_meters=self.radius,
            mode=self.mode,
        )


class GlobalSearchFiltersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    categories: list[str] = Field(default_factory=list, max_length=20)
    price_ranges: list[str] = Field(default_factory=list, max_length=10)
    tags: list[str] = Field(default_factory=list, max_length=30)
    published_from: datetime | None = None
    published_to: datetime | None = None
    event_from: datetime | None = None
    event_to: datetime | None = None
    club_mode: ClubAttendanceMode | None = None
    user_roles: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "GlobalSearchFiltersSchema":
        for field_name in (
            "published_from",
            "published_to",
            "event_from",
            "event_to",
        ):
            value = getattr(self, field_name)
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{field_name} must include a timezone")
        if (
            self.published_from is not None
            and self.published_to is not None
            and self.published_from > self.published_to
        ):
            raise ValueError("published_from must be before published_to")
        if (
            self.event_from is not None
            and self.event_to is not None
            and self.event_from > self.event_to
        ):
            raise ValueError("event_from must be before event_to")
        return self

    def to_domain(self) -> GlobalSearchFilters:
        return GlobalSearchFilters(
            city=self.city.strip() if self.city else None,
            state=self.state.strip() if self.state else None,
            categories=_clean_filter_values(self.categories),
            price_ranges=_clean_filter_values(self.price_ranges),
            tags=_clean_filter_values(self.tags),
            published_from=self.published_from,
            published_to=self.published_to,
            event_from=self.event_from,
            event_to=self.event_to,
            club_mode=self.club_mode,
            user_roles=_clean_filter_values(self.user_roles),
        )


class GlobalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    resource_types: list[SearchResourceType] | None = None
    per_type_limit: int = Field(default=5, ge=1, le=20)
    top_limit: int = Field(default=10, ge=1, le=50)
    requester_id: UUID | None = None
    cursors: dict[SearchResourceType, str] = Field(default_factory=dict)
    location: SearchLocationSchema | None = None
    filters: GlobalSearchFiltersSchema = Field(default_factory=GlobalSearchFiltersSchema)

    @model_validator(mode="after")
    def validate_cursors(self) -> "GlobalSearchRequest":
        if not self.cursors:
            return self
        if not self.resource_types:
            raise ValueError("resource_types is required when cursors are provided")
        requested = set(self.resource_types)
        unexpected = set(self.cursors) - requested
        if unexpected:
            names = ", ".join(sorted(resource_type.value for resource_type in unexpected))
            raise ValueError(f"cursor resources were not requested: {names}")
        return self

    def to_domain_criteria(self, nearby_boost: float = 0.12) -> SearchCriteria:
        return SearchCriteria(
            filters=self.filters.to_domain(),
            location=self.location.to_domain() if self.location else None,
            nearby_boost=nearby_boost,
        )

    def cursor_context_payload(
        self,
        nearby_boost: float = 0.12,
        policy_version: str = "global-search-relevance-v1",
    ) -> dict[str, Any]:
        filters = self.filters.model_dump(mode="json")
        for key in ("categories", "price_ranges", "tags", "user_roles"):
            filters[key] = sorted(
                {
                    str(value).strip().casefold()
                    for value in filters[key]
                    if str(value).strip()
                }
            )
        for key in ("city", "state"):
            if filters[key]:
                filters[key] = str(filters[key]).strip().casefold()
        for key in ("published_from", "published_to", "event_from", "event_to"):
            value = getattr(self.filters, key)
            filters[key] = (
                value.astimezone(UTC).isoformat() if value is not None else None
            )
        return {
            "query": normalize_search_query(self.query),
            "per_type_limit": self.per_type_limit,
            "requester_id": str(self.requester_id) if self.requester_id else None,
            "location": (
                self.location.model_dump(mode="json") if self.location else None
            ),
            "filters": filters,
            "nearby_boost": nearby_boost,
            "policy_version": policy_version,
        }


class SearchHitSchema(BaseModel):
    id: str
    resource_type: SearchResourceType
    title: str
    subtitle: str | None = None
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    is_nearby: bool | None = None
    proximity_boost: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GlobalSearchMetadataSchema(BaseModel):
    strategy: str
    queried_resources: list[SearchResourceType]
    failed_resources: dict[str, str]
    embedding_computed_once: bool


class SearchSectionPaginationSchema(BaseModel):
    page_size: int
    returned_count: int
    has_more: bool
    next_cursor: str | None = None


class GlobalSearchResponse(BaseModel):
    query: str
    normalized_query: str
    top_results: list[SearchHitSchema]
    sections: dict[str, list[SearchHitSchema]]
    pagination: dict[str, SearchSectionPaginationSchema]
    metadata: GlobalSearchMetadataSchema


def result_to_schema(result: SearchAllResult) -> GlobalSearchResponse:
    return GlobalSearchResponse(
        query=result.query,
        normalized_query=result.normalized_query,
        top_results=[_hit_to_schema(hit) for hit in result.top_results],
        sections={
            resource_type.value: [_hit_to_schema(hit) for hit in hits]
            for resource_type, hits in result.sections.items()
        },
        pagination={
            resource_type.value: SearchSectionPaginationSchema(
                page_size=page.page_size,
                returned_count=page.returned_count,
                has_more=(
                    page.has_more
                    and page.next_offset is not None
                    and page.next_offset <= MAX_SEARCH_OFFSET
                ),
                next_cursor=(
                    encode_search_cursor(
                        resource_type,
                        result.cursor_context,
                        page.next_offset,
                    )
                    if page.next_offset is not None
                    and page.next_offset <= MAX_SEARCH_OFFSET
                    else None
                ),
            )
            for resource_type, page in result.pagination.items()
        },
        metadata=GlobalSearchMetadataSchema(
            strategy="parallel_hybrid_fasttext_full_text_rrf",
            queried_resources=list(result.sections),
            failed_resources={
                resource_type.value: error
                for resource_type, error in result.failed_resources.items()
            },
            embedding_computed_once=True,
        ),
    )


def _hit_to_schema(hit: SearchHit) -> SearchHitSchema:
    return SearchHitSchema(
        id=hit.id,
        resource_type=hit.resource_type,
        title=hit.title,
        subtitle=hit.subtitle,
        score=round(hit.score, 4),
        semantic_score=(round(hit.semantic_score, 4) if hit.semantic_score is not None else None),
        lexical_score=(round(hit.lexical_score, 4) if hit.lexical_score is not None else None),
        is_nearby=hit.is_nearby,
        proximity_boost=round(hit.proximity_boost, 4),
        metadata=hit.metadata,
    )


def _clean_filter_values(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
