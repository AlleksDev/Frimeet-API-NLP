import asyncio
from collections import Counter
from math import sqrt

from app.modules.posts.domain.clustering import ClusterRunResult, KMeansResult
from app.modules.posts.domain.ports.cluster_training_repository import ClusterTrainingRepository
from app.modules.posts.domain.ports.clusterer import PostClusterer


class TrainPostClustersUseCase:
    ALGORITHM_VERSION = "sklearn-minibatch-kmeans-v1"

    def __init__(
        self,
        repository: ClusterTrainingRepository,
        clusterer: PostClusterer,
        min_posts: int = 20,
        min_k: int = 2,
        max_k: int = 50,
        lookback_days: int = 90,
        min_cluster_size: int = 2,
        max_cluster_ratio: float = 0.70,
        auto_activate: bool = False,
    ) -> None:
        self._repository = repository
        self._clusterer = clusterer
        self._min_posts = min_posts
        self._min_k = min_k
        self._max_k = max_k
        self._lookback_days = lookback_days
        self._min_cluster_size = min_cluster_size
        self._max_cluster_ratio = max_cluster_ratio
        self._auto_activate = auto_activate

    async def execute(self) -> ClusterRunResult:
        embeddings = await self._repository.list_active_embeddings(self._lookback_days)
        if len(embeddings) < self._min_posts:
            raise ValueError(
                f"se requieren al menos {self._min_posts} posts para entrenar K-Means"
            )
        estimated = round(sqrt(len(embeddings) / 2))
        upper = min(self._max_k, len(embeddings) - 1)
        lower = min(self._min_k, upper)
        candidate_ks = sorted(
            {
                max(lower, min(upper, value))
                for value in (estimated - 4, estimated, estimated + 4)
            }
        )
        run_id = await self._repository.create_run(
            planned_k=candidate_ks[len(candidate_ks) // 2],
            sample_size=len(embeddings),
            algorithm_version=self.ALGORITHM_VERSION,
            parameters={
                "candidate_ks": ",".join(str(value) for value in candidate_ks),
                "min_cluster_size": self._min_cluster_size,
                "max_cluster_ratio": self._max_cluster_ratio,
            },
        )
        try:
            evaluated: list[tuple[int, KMeansResult]] = []
            for k in candidate_ks:
                result = await asyncio.to_thread(self._clusterer.fit, embeddings, k)
                if _is_valid_distribution(
                    result,
                    len(embeddings),
                    self._min_cluster_size,
                    self._max_cluster_ratio,
                ):
                    evaluated.append((k, result))
            if not evaluated:
                raise ValueError("ningun K produjo una distribucion de clusters valida")
            k, result = max(
                evaluated,
                key=lambda item: (
                    item[1].silhouette_score
                    if item[1].silhouette_score is not None
                    else float("-inf"),
                    -item[1].inertia,
                ),
            )
            await self._repository.save_validated_run(run_id, k, result)
            status = "validated"
            if self._auto_activate:
                await self._repository.activate_run(run_id)
                status = "active"
            return ClusterRunResult(
                run_id,
                k,
                len(embeddings),
                result.inertia,
                result.silhouette_score,
                status,
            )
        except Exception as exc:
            await self._repository.mark_run_failed(run_id, str(exc))
            raise


def _is_valid_distribution(
    result: KMeansResult,
    sample_size: int,
    min_cluster_size: int,
    max_cluster_ratio: float,
) -> bool:
    counts = Counter(item.cluster_id for item in result.assignments)
    if not counts or min(counts.values()) < min_cluster_size:
        return False
    return max(counts.values()) / sample_size <= max_cluster_ratio
