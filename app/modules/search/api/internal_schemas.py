from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.modules.search.api.cursor import MAX_SEARCH_OFFSET, encode_search_cursor
from app.modules.search.api.schemas import (
    GlobalSearchFiltersSchema,
    SearchLocationSchema,
)
from app.modules.search.domain.filters import SearchCriteria
from app.modules.search.domain.models import (
    SearchCandidate,
    SearchCandidatesResult,
    SearchResourceType,
)
from app.modules.search.domain.query import normalize_search_query


class InternalSearchCandidatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    resource_types: list[SearchResourceType] = Field(..., min_length=1, max_length=6)
    candidate_limit_per_type: int = Field(default=20, ge=1, le=500)
    top_limit: int = Field(default=30, ge=1, le=3000)
    cursors: dict[SearchResourceType, str] = Field(default_factory=dict)
    as_of: datetime
    location: SearchLocationSchema | None = None
    filters: GlobalSearchFiltersSchema = Field(default_factory=GlobalSearchFiltersSchema)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not normalize_search_query(value):
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> "InternalSearchCandidatesRequest":
        if self.as_of.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        if len(self.resource_types) != len(set(self.resource_types)):
            raise ValueError("resource_types must not contain duplicates")
        unexpected = set(self.cursors) - set(self.resource_types)
        if unexpected:
            names = ", ".join(sorted(item.value for item in unexpected))
            raise ValueError(f"cursor resources were not requested: {names}")
        return self

    def to_domain_criteria(self, nearby_boost: float) -> SearchCriteria:
        return SearchCriteria(
            filters=self.filters.to_domain(),
            location=self.location.to_domain() if self.location else None,
            nearby_boost=nearby_boost,
            event_active_at=self.as_of,
        )

    def cursor_context_payload(
        self,
        nearby_boost: float,
        policy_version: str,
    ) -> dict[str, object]:
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
            filters[key] = value.astimezone(UTC).isoformat() if value else None
        return {
            "query": normalize_search_query(self.query),
            "candidate_limit_per_type": self.candidate_limit_per_type,
            "top_limit": self.top_limit,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "location": self.location.model_dump(mode="json") if self.location else None,
            "filters": filters,
            "nearby_boost": nearby_boost,
            "policy_version": policy_version,
        }


class InternalSearchCandidateSchema(BaseModel):
    id: str
    resource_type: SearchResourceType
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None


class InternalSearchPaginationSchema(BaseModel):
    returned_count: int
    has_more: bool
    next_cursor: str | None = None


class InternalSearchMetadataSchema(BaseModel):
    strategy: str
    failed_resources: dict[str, str]
    threshold_policy_version: str


class InternalSearchCandidatesResponse(BaseModel):
    query: str
    sections: dict[str, list[InternalSearchCandidateSchema]]
    pagination: dict[str, InternalSearchPaginationSchema]
    metadata: InternalSearchMetadataSchema


def candidates_result_to_schema(
    result: SearchCandidatesResult,
) -> InternalSearchCandidatesResponse:
    return InternalSearchCandidatesResponse(
        query=result.query,
        sections={
            resource_type.value: [_candidate_to_schema(item) for item in items]
            for resource_type, items in result.sections.items()
        },
        pagination={
            resource_type.value: InternalSearchPaginationSchema(
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
        metadata=InternalSearchMetadataSchema(
            strategy="hybrid_thresholded_candidates_v2",
            failed_resources={
                resource_type.value: error
                for resource_type, error in result.failed_resources.items()
            },
            threshold_policy_version=result.policy_version,
        ),
    )


def _candidate_to_schema(item: SearchCandidate) -> InternalSearchCandidateSchema:
    return InternalSearchCandidateSchema(
        id=item.id,
        resource_type=item.resource_type,
        score=round(item.score, 6),
        semantic_score=(
            round(item.semantic_score, 6) if item.semantic_score is not None else None
        ),
        lexical_score=(
            round(item.lexical_score, 6) if item.lexical_score is not None else None
        ),
    )
