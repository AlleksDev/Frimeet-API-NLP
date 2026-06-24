from typing import Protocol, Sequence

from app.modules.posts.domain.models import PostCandidate


class PostRanker(Protocol):
    def rank(self, posts: Sequence[PostCandidate], limit: int) -> list[PostCandidate]:
        """Rank candidate posts returned by the vector store."""
