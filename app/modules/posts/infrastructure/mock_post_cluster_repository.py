from typing import Sequence

from app.modules.posts.application.ports.cluster_repository import PostClusterRepository
from app.modules.posts.domain.models import PostCluster


class MockPostClusterRepository(PostClusterRepository):
    async def list_clusters(self) -> Sequence[PostCluster]:
        return [
            PostCluster(
                id="cluster_food_weekend",
                label="Planes de comida para fin de semana",
                post_ids=["post_1", "post_3"],
                size=2,
                metadata={"generated_by": "offline_placeholder_job"},
            ),
            PostCluster(
                id="cluster_outdoors",
                label="Paseos y fotos",
                post_ids=["post_2"],
                size=1,
                metadata={"generated_by": "offline_placeholder_job"},
            ),
        ]
