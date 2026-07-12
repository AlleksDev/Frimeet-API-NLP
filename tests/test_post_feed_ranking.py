import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.modules.posts.infrastructure.api import internal_router
from app.modules.posts.application.use_cases.rank_user_feed import RankUserFeedUseCase
from app.modules.posts.domain.feed import FeedCandidate, PostFeedFeatures
from app.modules.posts.infrastructure.kmeans_clusterer import KMeansPostClusterer
from app.modules.posts.domain.clustering import PostEmbedding
from app.shared.config.settings import get_settings
from app.shared.vector_store.aws_pgvector import READ_CONTRACT_SIGNATURES


class FeatureRepositoryStub:
    async def get_features(self, user_id: str, post_ids: list[str]):
        return [PostFeedFeatures(post_id, 0.8, index % 2) for index, post_id in enumerate(post_ids)], "run-1", False


@pytest.mark.asyncio
async def test_rank_feed_returns_only_allowed_candidates() -> None:
    now = datetime.now(UTC)
    candidates = [FeedCandidate("p1", "u1", now, 1.0, 0.5), FeedCandidate("p2", "u2", now - timedelta(hours=2), 0.0, 0.2)]
    result = await RankUserFeedUseCase(FeatureRepositoryStub()).execute("viewer", candidates, now, 10)
    assert {item.post_id for item in result.items} == {"p1", "p2"}
    assert result.cluster_run_id == "run-1"


def test_internal_feed_endpoint_rejects_duplicate_ids() -> None:
    client = TestClient(create_app())
    candidate = {"post_id": "p1", "author_id": "u1", "created_at": "2026-07-04T12:00:00Z", "social_affinity": 0.5, "engagement_score": 0.5}
    token = get_settings().post_feed_internal_token or ""
    response = client.post("/internal/posts/feed/rank", headers={"Authorization": f"Bearer {token}"}, json={"user_id": "viewer", "candidate_posts": [candidate, candidate], "snapshot_at": "2026-07-04T13:00:00Z", "result_limit": 2})
    assert response.status_code == 422


@pytest.mark.parametrize("authorization", [None, "internal-secret", "Basic internal-secret"])
def test_internal_feed_endpoint_requires_strict_bearer(
    monkeypatch: pytest.MonkeyPatch, authorization: str | None
) -> None:
    monkeypatch.setattr(
        internal_router,
        "get_settings",
        lambda: SimpleNamespace(
            env="production", post_feed_internal_token="internal-secret"
        ),
    )
    headers = {"Authorization": authorization} if authorization else {}
    response = TestClient(create_app()).post(
        "/internal/posts/feed/rank",
        headers=headers,
        json={
            "user_id": "viewer",
            "candidate_posts": [],
            "snapshot_at": "2026-07-04T13:00:00Z",
            "result_limit": 0,
        },
    )
    assert response.status_code == 401


def test_internal_feed_endpoint_accepts_500_uuid_candidates() -> None:
    candidates = [
        {
            "post_id": f"00000000-0000-0000-0000-{index:012d}",
            "author_id": f"10000000-0000-0000-0000-{index:012d}",
            "created_at": "2026-07-04T12:00:00Z",
            "social_affinity": 0.5,
            "engagement_score": 0.5,
        }
        for index in range(500)
    ]
    payload = {
        "user_id": "20000000-0000-0000-0000-000000000001",
        "candidate_posts": candidates,
        "snapshot_at": "2026-07-04T13:00:00Z",
        "result_limit": 500,
    }
    assert len(json.dumps(payload).encode("utf-8")) > 65_536
    settings = get_settings()
    token = settings.post_feed_internal_token or ""
    response = TestClient(create_app()).post(
        "/internal/posts/feed/rank",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 500


def test_ready_contract_includes_feed_features() -> None:
    assert READ_CONTRACT_SIGNATURES["get_post_feed_features"] == (
        "get_post_feed_features(text, text[])"
    )


def test_minibatch_kmeans_is_reproducible() -> None:
    embeddings = [PostEmbedding(f"p{i}", [float(i % 2), float((i + 1) % 2)]) for i in range(6)]
    clusterer = KMeansPostClusterer(random_state=7)
    first = clusterer.fit(embeddings, 2)
    second = clusterer.fit(embeddings, 2)
    assert first.centroids == second.centroids
    assert first.assignments == second.assignments


class PartialFeatureRepositoryStub:
    async def get_features(self, user_id: str, post_ids: list[str]):
        return [PostFeedFeatures(post_ids[0], 0.9, 1)], "run-2", False


@pytest.mark.asyncio
async def test_rank_feed_keeps_candidates_without_embedding() -> None:
    now = datetime.now(UTC)
    candidates = [
        FeedCandidate("p1", "a1", now, 0.0, 0.0),
        FeedCandidate("p2", "a2", now, 1.0, 0.5),
    ]
    result = await RankUserFeedUseCase(PartialFeatureRepositoryStub()).execute(
        "viewer", candidates, now, 10
    )
    assert {item.post_id for item in result.items} == {"p1", "p2"}
    assert next(item for item in result.items if item.post_id == "p2").cluster_id is None
