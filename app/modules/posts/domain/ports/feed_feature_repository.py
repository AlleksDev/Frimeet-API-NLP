from typing import Protocol, Sequence

from app.modules.posts.domain.feed import PostFeedFeatures


class FeedFeatureRepository(Protocol):
    async def get_features(
        self, user_id: str, post_ids: Sequence[str]
    ) -> tuple[list[PostFeedFeatures], str | None, bool]:
        """Return derived technical features, active cluster run and cold-start flag."""
