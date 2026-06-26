from typing import Protocol, Sequence

from app.modules.posts.domain.models import PostCluster


class PostClusterRepository(Protocol):
    async def list_clusters(self) -> Sequence[PostCluster]:
        """Read previously generated post clusters."""
