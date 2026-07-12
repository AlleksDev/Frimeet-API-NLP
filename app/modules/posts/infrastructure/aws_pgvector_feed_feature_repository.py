from collections.abc import Sequence

from app.modules.posts.domain.feed import PostFeedFeatures
from app.modules.posts.domain.ports.feed_feature_repository import FeedFeatureRepository
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


class AwsPgvectorFeedFeatureRepository(FeedFeatureRepository):
    def __init__(self, client: AwsPgvectorClient) -> None:
        self._client = client

    async def get_features(
        self, user_id: str, post_ids: Sequence[str]
    ) -> tuple[list[PostFeedFeatures], str | None, bool]:
        if not post_ids:
            return [], None, True
        query = "SELECT * FROM get_post_feed_features($1::text, $2::text[])"
        async with self._client.connection() as connection:
            rows = await connection.fetch(query, user_id, list(post_ids))
        features = [
            PostFeedFeatures(
                post_id=str(row["external_id"]),
                semantic_similarity=float(row["semantic_similarity"] or 0.0),
                cluster_id=int(row["cluster_id"]) if row["cluster_id"] is not None else None,
                embedding=(
                    tuple(_parse_vector(row["post_embedding"]))
                    if row["post_embedding"] is not None
                    else None
                ),
            )
            for row in rows
        ]
        cluster_run_id = next(
            (str(row["cluster_run_id"]) for row in rows if row["cluster_run_id"]), None
        )
        cold_start = all(bool(row["cold_start"]) for row in rows) if rows else True
        return features, cluster_run_id, cold_start


def _parse_vector(value: object) -> list[float]:
    return [float(item) for item in str(value).strip("[]").split(",") if item]
