from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FeedCandidate:
    post_id: str
    author_id: str
    created_at: datetime
    social_affinity: float
    engagement_score: float
    author_affinity: float = 0.0


@dataclass(frozen=True)
class PostFeedFeatures:
    post_id: str
    semantic_similarity: float
    cluster_id: int | None
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class RankedFeedPost:
    post_id: str
    score: float
    cluster_id: int | None
    author_id: str
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FeedRanking:
    items: list[RankedFeedPost]
    ranking_version: str
    cluster_run_id: str | None
    cold_start: bool
    missing_embedding_count: int = 0
    diversity_relaxations: int = 0
    duplicate_penalized_count: int = 0
