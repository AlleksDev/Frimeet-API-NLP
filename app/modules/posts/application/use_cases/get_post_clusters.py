from dataclasses import dataclass, field
from typing import Any

from app.modules.posts.application.ports.cluster_repository import PostClusterRepository
from app.modules.posts.domain.models import PostCluster


@dataclass(frozen=True)
class GetPostClustersResult:
    clusters: list[PostCluster]
    metadata: dict[str, Any] = field(default_factory=dict)


class GetPostClustersUseCase:
    def __init__(self, cluster_repository: PostClusterRepository) -> None:
        self._cluster_repository = cluster_repository

    async def execute(self) -> GetPostClustersResult:
        clusters = list(await self._cluster_repository.list_clusters())
        return GetPostClustersResult(
            clusters=clusters,
            metadata={
                "source": "offline_job_output",
                "computed_during_request": False,
            },
        )
