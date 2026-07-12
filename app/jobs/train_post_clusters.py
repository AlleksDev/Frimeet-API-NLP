import asyncio

from app.modules.posts.application.use_cases.train_post_clusters import TrainPostClustersUseCase
from app.modules.posts.infrastructure.aws_pgvector_cluster_training_repository import (
    AwsPgvectorClusterTrainingRepository,
)
from app.modules.posts.infrastructure.kmeans_clusterer import MiniBatchKMeansPostClusterer
from app.shared.config.settings import get_settings
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


async def main() -> None:
    settings = get_settings()
    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("train_post_clusters requiere VECTOR_STORE_PROVIDER=aws_pgvector")
    repository = AwsPgvectorClusterTrainingRepository(
        AwsPgvectorClient(settings, role="writer"), settings
    )
    result = await TrainPostClustersUseCase(
        repository,
        MiniBatchKMeansPostClusterer(
            random_state=settings.kmeans_random_state,
            batch_size=settings.kmeans_batch_size,
        ),
        min_posts=settings.kmeans_min_posts,
        min_k=settings.kmeans_min_k,
        max_k=settings.kmeans_max_k,
        lookback_days=settings.kmeans_lookback_days,
        min_cluster_size=settings.kmeans_min_cluster_size,
        max_cluster_ratio=settings.kmeans_max_cluster_ratio,
        auto_activate=settings.kmeans_auto_activate,
    ).execute()
    print(
        f"cluster_run={result.run_id} k={result.k} "
        f"sample_size={result.sample_size} inertia={result.inertia:.6f} "
        f"status={result.status}"
    )


if __name__ == "__main__":
    asyncio.run(main())
