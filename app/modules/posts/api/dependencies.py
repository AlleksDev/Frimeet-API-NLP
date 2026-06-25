from functools import lru_cache

from app.modules.posts.application.use_cases.get_post_clusters import GetPostClustersUseCase
from app.modules.posts.application.use_cases.recommend_posts import RecommendPostsUseCase
from app.modules.posts.infrastructure.aws_pgvector_post_repository import (
    AwsPgvectorPostRepository,
)
from app.modules.posts.infrastructure.chroma_post_repository import ChromaPostVectorRepository
from app.modules.posts.infrastructure.mock_post_cluster_repository import MockPostClusterRepository
from app.modules.posts.infrastructure.mock_post_repository import MockPostVectorRepository
from app.modules.posts.infrastructure.simple_post_ranker import SimplePostRanker
from app.shared.chroma.client import ChromaHttpClientFactory
from app.shared.chroma.vector_store import ChromaVectorStore
from app.shared.config.settings import get_settings
from app.shared.dependencies import get_embedding_provider
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


@lru_cache
def get_post_repository() -> (
    MockPostVectorRepository | ChromaPostVectorRepository | AwsPgvectorPostRepository
):
    settings = get_settings()
    if settings.vector_store_provider == "aws_pgvector":
        return AwsPgvectorPostRepository(vector_client=AwsPgvectorClient(settings))
    if settings.vector_store_mode == "chroma":
        return ChromaPostVectorRepository(
            vector_store=ChromaVectorStore(
                client_factory=ChromaHttpClientFactory(settings),
                collection_name=settings.chroma_posts_collection,
            )
        )
    return MockPostVectorRepository(embedding_provider=get_embedding_provider())


@lru_cache
def get_post_ranker() -> SimplePostRanker:
    return SimplePostRanker()


@lru_cache
def get_post_cluster_repository() -> MockPostClusterRepository:
    return MockPostClusterRepository()


@lru_cache
def get_recommend_posts_use_case() -> RecommendPostsUseCase:
    return RecommendPostsUseCase(
        embedding_provider=get_embedding_provider(),
        post_repository=get_post_repository(),
        ranker=get_post_ranker(),
    )


@lru_cache
def get_post_clusters_use_case() -> GetPostClustersUseCase:
    return GetPostClustersUseCase(cluster_repository=get_post_cluster_repository())
