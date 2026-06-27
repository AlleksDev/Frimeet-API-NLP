from functools import lru_cache

from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.evaluate_place_search import (
    EvaluatePlaceSearchUseCase,
)
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.infrastructure.aws_pgvector_place_repository import (
    AwsPgvectorPlaceRepository,
)
from app.modules.places.infrastructure.main_api_nearby_place_provider import (
    MainApiNearbyPlaceProvider,
)
from app.modules.places.infrastructure.mock_place_repository import MockPlaceVectorRepository
from app.modules.places.infrastructure.place_search_benchmark import (
    BENCHMARK_NAME,
    QRELS_SOURCE,
    get_default_place_search_benchmark,
)
from app.modules.places.infrastructure.tfidf_place_ranker import TfidfPlaceRanker
from app.shared.cache.memory import SimpleTTLCache
from app.shared.config.settings import get_settings
from app.shared.dependencies import get_embedding_provider, get_llm_provider
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


@lru_cache
def get_place_repository() -> MockPlaceVectorRepository | AwsPgvectorPlaceRepository:
    settings = get_settings()
    if settings.vector_store_provider == "aws_pgvector":
        return AwsPgvectorPlaceRepository(vector_client=AwsPgvectorClient(settings, role="reader"))
    return MockPlaceVectorRepository(embedding_provider=get_embedding_provider())


@lru_cache
def get_place_ranker() -> TfidfPlaceRanker:
    return TfidfPlaceRanker()


@lru_cache
def get_place_search_cache() -> SimpleTTLCache:
    settings = get_settings()
    return SimpleTTLCache(default_ttl_seconds=settings.vector_search_cache_ttl_seconds)


@lru_cache
def get_nearby_place_provider() -> MainApiNearbyPlaceProvider:
    return MainApiNearbyPlaceProvider(settings=get_settings())


@lru_cache
def get_search_places_use_case() -> SearchPlacesUseCase:
    return SearchPlacesUseCase(
        embedding_provider=get_embedding_provider(),
        place_repository=get_place_repository(),
        ranker=get_place_ranker(),
        cache=get_place_search_cache(),
        nearby_place_provider=get_nearby_place_provider(),
    )


@lru_cache
def get_recommend_places_use_case() -> RecommendPlacesUseCase:
    return RecommendPlacesUseCase(
        search_use_case=get_search_places_use_case(),
        evaluation_use_case=get_evaluate_place_search_use_case(),
        llm_provider=get_llm_provider(),
        output_guard=PlaceChatOutputGuard(),
    )


@lru_cache
def get_evaluate_place_search_use_case() -> EvaluatePlaceSearchUseCase:
    settings = get_settings()
    embedding_provider = MockEmbeddingProvider(dimension=settings.embedding_dimension)
    benchmark_search = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=TfidfPlaceRanker(),
    )
    return EvaluatePlaceSearchUseCase(
        search_use_case=benchmark_search,
        cases=get_default_place_search_benchmark(),
        benchmark=BENCHMARK_NAME,
        qrels_source=QRELS_SOURCE,
    )


@lru_cache
def get_chat_places_use_case() -> ChatPlacesUseCase:
    return ChatPlacesUseCase(
        search_use_case=get_search_places_use_case(),
        llm_provider=get_llm_provider(),
        output_guard=PlaceChatOutputGuard(),
    )
