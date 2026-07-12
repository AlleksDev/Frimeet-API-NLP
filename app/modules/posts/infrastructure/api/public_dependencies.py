from functools import lru_cache

from app.modules.posts.application.use_cases.recommend_posts import RecommendPostsUseCase
from app.modules.posts.infrastructure.aws_pgvector_post_repository import AwsPgvectorPostRepository
from app.modules.posts.infrastructure.mock_post_repository import MockPostVectorRepository
from app.modules.posts.infrastructure.simple_post_ranker import SimplePostRanker
from app.modules.posts.infrastructure.text_preprocessor import SharedTextPreprocessor
from app.shared.cache.memory import SimpleTTLCache
from app.shared.config.settings import get_settings
from app.shared.dependencies import get_embedding_provider
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


@lru_cache
def get_post_repository() -> MockPostVectorRepository | AwsPgvectorPostRepository:
    settings = get_settings()
    if settings.vector_store_provider == "aws_pgvector":
        return AwsPgvectorPostRepository(AwsPgvectorClient(settings, role="reader"))
    return MockPostVectorRepository(get_embedding_provider())


@lru_cache
def get_recommend_posts_use_case() -> RecommendPostsUseCase:
    settings = get_settings()
    return RecommendPostsUseCase(
        embedding_provider=get_embedding_provider(),
        text_preprocessor=SharedTextPreprocessor(),
        post_repository=get_post_repository(),
        ranker=SimplePostRanker(),
        cache=SimpleTTLCache(settings.vector_search_cache_ttl_seconds),
    )
