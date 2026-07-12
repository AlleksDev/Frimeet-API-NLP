from functools import lru_cache

from app.modules.posts.application.use_cases.rank_user_feed import RankUserFeedUseCase
from app.modules.posts.application.use_cases.activate_post_cluster_run import ActivatePostClusterRunUseCase
from app.modules.posts.application.use_cases.get_post_cluster_status import GetPostClusterStatusUseCase
from app.modules.posts.application.use_cases.get_post_cluster_run import GetPostClusterRunUseCase
from app.modules.posts.application.use_cases.train_post_clusters import TrainPostClustersUseCase
from app.modules.posts.infrastructure.aws_pgvector_cluster_training_repository import AwsPgvectorClusterTrainingRepository
from app.modules.posts.infrastructure.aws_pgvector_feed_feature_repository import (
    AwsPgvectorFeedFeatureRepository,
)
from app.modules.posts.infrastructure.mock_feed_feature_repository import (
    MockFeedFeatureRepository,
)
from app.shared.config.settings import get_settings
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


@lru_cache
def get_rank_user_feed_use_case() -> RankUserFeedUseCase:
    settings = get_settings()
    if settings.vector_store_provider == "aws_pgvector":
        repository = AwsPgvectorFeedFeatureRepository(
            AwsPgvectorClient(settings, role="reader")
        )
    else:
        repository = MockFeedFeatureRepository()
    return RankUserFeedUseCase(
        repository,
        duplicate_similarity_threshold=settings.feed_duplicate_similarity_threshold,
        duplicate_penalty=settings.feed_duplicate_penalty,
    )


@lru_cache
def get_cluster_repository() -> AwsPgvectorClusterTrainingRepository:
    settings = get_settings()
    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("la operacion de clusters requiere aws_pgvector")
    return AwsPgvectorClusterTrainingRepository(
        AwsPgvectorClient(settings, role="writer"), settings
    )


def get_train_post_clusters_use_case() -> TrainPostClustersUseCase:
    from app.modules.posts.infrastructure.kmeans_clusterer import (
        MiniBatchKMeansPostClusterer,
    )

    settings = get_settings()
    return TrainPostClustersUseCase(
        get_cluster_repository(),
        MiniBatchKMeansPostClusterer(
            settings.kmeans_random_state, settings.kmeans_batch_size
        ),
        min_posts=settings.kmeans_min_posts,
        min_k=settings.kmeans_min_k,
        max_k=settings.kmeans_max_k,
        lookback_days=settings.kmeans_lookback_days,
        min_cluster_size=settings.kmeans_min_cluster_size,
        max_cluster_ratio=settings.kmeans_max_cluster_ratio,
        auto_activate=settings.kmeans_auto_activate,
    )


def get_cluster_status_use_case() -> GetPostClusterStatusUseCase:
    settings = get_settings()
    repository = AwsPgvectorClusterTrainingRepository(
        AwsPgvectorClient(settings, role="reader"), settings
    )
    return GetPostClusterStatusUseCase(repository)


def get_cluster_run_use_case() -> GetPostClusterRunUseCase:
    settings = get_settings()
    repository = AwsPgvectorClusterTrainingRepository(
        AwsPgvectorClient(settings, role="reader"), settings
    )
    return GetPostClusterRunUseCase(repository)


def get_activate_cluster_use_case() -> ActivatePostClusterRunUseCase:
    return ActivatePostClusterRunUseCase(get_cluster_repository())
