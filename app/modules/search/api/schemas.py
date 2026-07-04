from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.modules.search.domain.models import SearchAllResult, SearchHit, SearchResourceType


class GlobalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    resource_types: list[SearchResourceType] | None = None
    per_type_limit: int = Field(default=5, ge=1, le=20)
    top_limit: int = Field(default=10, ge=1, le=50)
    requester_id: UUID | None = None


class SearchHitSchema(BaseModel):
    id: str
    resource_type: SearchResourceType
    title: str
    subtitle: str | None = None
    score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GlobalSearchMetadataSchema(BaseModel):
    strategy: str
    queried_resources: list[SearchResourceType]
    failed_resources: dict[str, str]
    embedding_computed_once: bool


class GlobalSearchResponse(BaseModel):
    query: str
    normalized_query: str
    top_results: list[SearchHitSchema]
    sections: dict[str, list[SearchHitSchema]]
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
        metadata=hit.metadata,
    )
