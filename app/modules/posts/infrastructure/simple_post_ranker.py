from typing import Sequence

from app.modules.posts.application.ports.ranker import PostRanker
from app.modules.posts.domain.models import PostCandidate


class SimplePostRanker(PostRanker):
    def rank(self, posts: Sequence[PostCandidate], limit: int) -> list[PostCandidate]:
        return sorted(posts, key=lambda post: post.score, reverse=True)[:limit]
