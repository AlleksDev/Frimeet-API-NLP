from datetime import datetime
from hashlib import sha256
from math import exp, sqrt

from app.modules.posts.domain.feed import (
    FeedCandidate,
    FeedRanking,
    RankedFeedPost,
)
from app.modules.posts.domain.ports.feed_feature_repository import FeedFeatureRepository


class RankUserFeedUseCase:
    RANKING_VERSION = "feed-v1"

    def __init__(
        self,
        feature_repository: FeedFeatureRepository,
        duplicate_similarity_threshold: float = 0.92,
        duplicate_penalty: float = 0.15,
    ) -> None:
        self._feature_repository = feature_repository
        self._duplicate_similarity_threshold = duplicate_similarity_threshold
        self._duplicate_penalty = duplicate_penalty

    async def execute(
        self,
        user_id: str,
        candidates: list[FeedCandidate],
        snapshot_at: datetime,
        result_limit: int,
    ) -> FeedRanking:
        allowed = {candidate.post_id: candidate for candidate in candidates}
        features, cluster_run_id, cold_start = await self._feature_repository.get_features(
            user_id, list(allowed)
        )
        feature_by_post = {feature.post_id: feature for feature in features}
        scored: list[RankedFeedPost] = []
        for candidate in candidates:
            feature = feature_by_post.get(candidate.post_id)
            age_hours = max(
                0.0, (snapshot_at - candidate.created_at).total_seconds() / 3600.0
            )
            freshness = exp(-age_hours / 72.0)
            semantic = (
                feature.semantic_similarity
                if feature is not None and not cold_start
                else 0.0
            )
            cluster_id = feature.cluster_id if feature is not None else None
            exploration = _stable_exploration(
                user_id, candidate.post_id, snapshot_at
            )
            score = (
                0.40 * semantic
                + 0.20 * candidate.social_affinity
                + 0.15 * freshness
                + 0.10 * candidate.engagement_score
                + 0.10 * candidate.author_affinity
                + 0.05 * exploration
            )
            scored.append(
                RankedFeedPost(
                    candidate.post_id,
                    score,
                    cluster_id,
                    candidate.author_id,
                    feature.embedding if feature is not None else None,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.post_id))
        scored, duplicate_penalized = _penalize_semantic_duplicates(
            scored,
            self._duplicate_similarity_threshold,
            self._duplicate_penalty,
        )
        scored.sort(key=lambda item: (-item.score, item.post_id))
        diversified, relaxations = _diversify(scored, max(0, result_limit))
        return FeedRanking(
            items=diversified,
            ranking_version=self.RANKING_VERSION,
            cluster_run_id=cluster_run_id,
            cold_start=cold_start,
            missing_embedding_count=sum(
                1
                for candidate in candidates
                if feature_by_post.get(candidate.post_id) is None
                or feature_by_post[candidate.post_id].embedding is None
            ),
            diversity_relaxations=relaxations,
            duplicate_penalized_count=duplicate_penalized,
        )


def _stable_exploration(user_id: str, post_id: str, snapshot_at: datetime) -> float:
    time_bucket = snapshot_at.strftime("%Y-%m-%d")
    digest = sha256(f"{user_id}:{post_id}:{time_bucket}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _penalize_semantic_duplicates(
    items: list[RankedFeedPost], threshold: float, penalty: float
) -> tuple[list[RankedFeedPost], int]:
    result: list[RankedFeedPost] = []
    penalized = 0
    for item in items:
        similarities = [
            _cosine(item.embedding, previous.embedding)
            for previous in result
            if item.embedding is not None and previous.embedding is not None
        ]
        max_similarity = max(similarities, default=-1.0)
        score = item.score
        if max_similarity >= threshold:
            score -= penalty * max_similarity
            penalized += 1
        result.append(
            RankedFeedPost(
                item.post_id,
                score,
                item.cluster_id,
                item.author_id,
                item.embedding,
            )
        )
    return result, penalized


def _cosine(
    left: tuple[float, ...] | None, right: tuple[float, ...] | None
) -> float:
    if left is None or right is None or len(left) != len(right) or not left:
        return -1.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _diversify(
    items: list[RankedFeedPost], limit: int
) -> tuple[list[RankedFeedPost], int]:
    pending = list(items)
    result: list[RankedFeedPost] = []
    cluster_counts: dict[int, int] = {}
    relaxations = 0
    while pending and len(result) < limit:
        selected_index = _find_candidate_index(
            pending, result, cluster_counts, enforce_page_limit=True
        )
        if selected_index is None:
            selected_index = _find_candidate_index(
                pending, result, cluster_counts, enforce_page_limit=False
            )
            if selected_index is not None:
                relaxations += 1
        if selected_index is None:
            # The constraints are impossible with the remaining candidates. Relax
            # deterministically instead of silently dropping allowed posts.
            selected_index = 0
            relaxations += 1
        selected = pending.pop(selected_index)
        result.append(selected)
        if selected.cluster_id is not None:
            cluster_counts[selected.cluster_id] = cluster_counts.get(selected.cluster_id, 0) + 1
    return result, relaxations


def _find_candidate_index(
    pending: list[RankedFeedPost],
    result: list[RankedFeedPost],
    cluster_counts: dict[int, int],
    *,
    enforce_page_limit: bool,
) -> int | None:
    for index, item in enumerate(pending):
        same_cluster = (
            item.cluster_id is not None
            and len(result) >= 2
            and result[-1].cluster_id == item.cluster_id
            and result[-2].cluster_id == item.cluster_id
        )
        same_author = (
            len(result) >= 2
            and result[-1].author_id == item.author_id
            and result[-2].author_id == item.author_id
        )
        page_limit_reached = (
            enforce_page_limit
            and item.cluster_id is not None
            and cluster_counts.get(item.cluster_id, 0) >= 4
        )
        if not same_cluster and not same_author and not page_limit_reached:
            return index
    return None
