from collections.abc import Sequence

from app.modules.posts.domain.feed import PostFeedFeatures
from app.modules.posts.domain.ports.feed_feature_repository import FeedFeatureRepository


class MockFeedFeatureRepository(FeedFeatureRepository):
    async def get_features(
        self, user_id: str, post_ids: Sequence[str]
    ) -> tuple[list[PostFeedFeatures], str | None, bool]:
        return (
            [
                PostFeedFeatures(
                    post_id=post_id,
                    semantic_similarity=(sum(ord(char) for char in post_id) % 100) / 100,
                    cluster_id=index % 4,
                )
                for index, post_id in enumerate(post_ids)
            ],
            "00000000-0000-0000-0000-000000000001",
            False,
        )
