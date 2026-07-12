from typing import Protocol

from app.modules.posts.domain.clustering import KMeansResult, PostEmbedding


class ClusterTrainingRepository(Protocol):
    async def list_active_embeddings(self, lookback_days: int) -> list[PostEmbedding]: ...

    async def create_run(
        self,
        planned_k: int,
        sample_size: int,
        algorithm_version: str,
        parameters: dict[str, int | float | str],
    ) -> str: ...

    async def save_validated_run(
        self,
        run_id: str,
        k: int,
        result: KMeansResult,
    ) -> None: ...

    async def mark_run_failed(self, run_id: str, error_message: str) -> None: ...
    async def activate_run(self, run_id: str) -> None: ...
