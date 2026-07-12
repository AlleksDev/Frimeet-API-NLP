from app.modules.posts.domain.clustering import ClusterStatus
from app.modules.posts.domain.ports.cluster_operations_repository import ClusterOperationsRepository


class GetPostClusterStatusUseCase:
    def __init__(self, repository: ClusterOperationsRepository) -> None:
        self._repository = repository

    async def execute(self) -> ClusterStatus:
        return await self._repository.get_status()
