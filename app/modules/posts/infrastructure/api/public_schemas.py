from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.modules.posts.domain.models import PostCandidate


class PostRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, max_length=500)
    city: str | None = Field(default=None, max_length=80)
    limit: int = Field(default=10, ge=1, le=20)


class PostResultSchema(BaseModel):
    id: str
    title: str
    score: float
    city: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PostRecommendationResponse(BaseModel):
    query: str
    posts: list[PostResultSchema]
    metadata: dict[str, Any] = Field(default_factory=dict)


def post_to_schema(post: PostCandidate) -> PostResultSchema:
    return PostResultSchema(
        id=post.id,
        title=post.title,
        score=round(post.score, 4),
        city=post.city,
        tags=post.tags,
        metadata=post.metadata,
    )
