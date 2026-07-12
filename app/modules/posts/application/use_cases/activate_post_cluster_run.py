from app.modules.posts.domain.ports.cluster_operations_repository import ClusterOperationsRepository


class ActivatePostClusterRunUseCase:
    def __init__(self, repository: ClusterOperationsRepository) -> None:
        self._repository = repository

    async def execute(self, run_id: str) -> None:
        await self._repository.activate_run(run_id)
