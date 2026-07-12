from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.modules.posts.domain.feed import FeedCandidate, FeedRanking


class FeedCandidateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    post_id: str = Field(min_length=1, max_length=100)
    author_id: str = Field(min_length=1, max_length=100)
    created_at: datetime
    social_affinity: float = Field(ge=0, le=1)
    engagement_score: float = Field(ge=0, le=1)
    author_affinity: float = Field(default=0.0, ge=0, le=1)


class RankFeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    candidate_posts: list[FeedCandidateSchema] = Field(max_length=500)
    snapshot_at: datetime
    result_limit: int = Field(ge=0, le=500)


class RankedFeedItemSchema(BaseModel):
    post_id: str
    score: float
    cluster_id: int | None = None


class RankFeedResponse(BaseModel):
    items: list[RankedFeedItemSchema]
    ranking_version: str
    cluster_run_id: str | None = None
    cold_start: bool
    missing_embedding_count: int = 0
    diversity_relaxations: int = 0
    duplicate_penalized_count: int = 0


class ClusterRunResponse(BaseModel):
    run_id: str
    k: int
    sample_size: int
    inertia: float
    silhouette_score: float | None = None
    status: str


class ClusterTrainingScheduledResponse(BaseModel):
    status: str = "scheduled"
    message: str = "entrenamiento enviado al worker en background"


class ClusterStatusResponse(BaseModel):
    active_run_id: str | None = None
    k: int | None = None
    sample_size: int | None = None
    silhouette_score: float | None = None
    activated_at: str | None = None


class ClusterActivationResponse(BaseModel):
    run_id: str
    status: str = "active"


class ClusterRunDetailResponse(BaseModel):
    run_id: str
    status: str
    k: int
    sample_size: int
    inertia: float | None = None
    silhouette_score: float | None = None
    started_at: str
    completed_at: str | None = None
    activated_at: str | None = None
    error_message: str | None = None


def to_domain(candidate: FeedCandidateSchema) -> FeedCandidate:
    return FeedCandidate(**candidate.model_dump())


def to_response(ranking: FeedRanking) -> RankFeedResponse:
    return RankFeedResponse(
        items=[
            RankedFeedItemSchema(
                post_id=item.post_id,
                score=item.score,
                cluster_id=item.cluster_id,
            )
            for item in ranking.items
        ],
        ranking_version=ranking.ranking_version,
        cluster_run_id=ranking.cluster_run_id,
        cold_start=ranking.cold_start,
        missing_embedding_count=ranking.missing_embedding_count,
        diversity_relaxations=ranking.diversity_relaxations,
        duplicate_penalized_count=ranking.duplicate_penalized_count,
    )
