from typing import Protocol, Sequence

from app.modules.posts.domain.models import PostCandidate


class PostVectorRepository(Protocol):
    async def search(
        self,
        embedding: list[float],
        city: str | None,
        limit: int,
    ) -> Sequence[PostCandidate]:
        """Search post vectors using an already generated query embedding."""
