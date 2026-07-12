from app.modules.posts.domain.clustering import ClusterRunDetail
from app.modules.posts.domain.ports.cluster_operations_repository import ClusterOperationsRepository


class GetPostClusterRunUseCase:
    def __init__(self, repository: ClusterOperationsRepository) -> None:
        self._repository = repository

    async def execute(self, run_id: str) -> ClusterRunDetail | None:
        return await self._repository.get_run(run_id)
