from functools import lru_cache

from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsUseCase,
)
from app.modules.places.application.use_cases.evaluate_place_search import (
    EvaluatePlaceSearchUseCase,
)
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.infrastructure.aws_pgvector_place_repository import (
    AwsPgvectorPlaceRepository,
)
from app.modules.places.infrastructure.bm25_place_ranker import Bm25PlaceRanker
from app.modules.places.infrastructure.main_api_nearby_place_provider import (
    MainApiNearbyPlaceProvider,
)
from app.modules.places.infrastructure.deterministic_intent_parser import (
    DeterministicPlaceChatIntentParser,
    load_place_chat_taxonomy,
)
from app.modules.places.infrastructure.hybrid_chat_retriever import (
    HybridContentPlaceChatRetriever,
)
from app.modules.places.infrastructure.main_api_place_anchor_resolver import (
    MainApiPlaceAnchorResolver,
    MockPlaceAnchorResolver,
)
from app.modules.places.infrastructure.mock_place_repository import MockPlaceVectorRepository
from app.modules.places.infrastructure.place_search_benchmark import (
    BENCHMARK_NAME,
    QRELS_SOURCE,
    get_default_place_search_benchmark,
)
from app.modules.places.infrastructure.semantic_place_ranker import SemanticPlaceRanker
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
def get_place_ranker() -> SemanticPlaceRanker:
    settings = get_settings()
    return SemanticPlaceRanker(dimension=settings.embedding_dimension)


@lru_cache
def get_place_search_cache() -> SimpleTTLCache:
    settings = get_settings()
    return SimpleTTLCache(default_ttl_seconds=settings.vector_search_cache_ttl_seconds)


@lru_cache
def get_place_chat_cache() -> SimpleTTLCache:
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
        relevance_threshold=get_settings().semantic_relevance_threshold,
        no_match_threshold=get_settings().semantic_no_match_threshold,
    )


@lru_cache
def get_recommend_places_use_case() -> RecommendPlacesUseCase:
    return RecommendPlacesUseCase(
        search_use_case=get_search_places_use_case(),
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
        ranker=Bm25PlaceRanker(k1=settings.bm25_k1, b=settings.bm25_b),
        relevance_threshold=settings.bm25_relevance_threshold,
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


@lru_cache
def get_place_chat_intent_parser() -> DeterministicPlaceChatIntentParser:
    settings = get_settings()
    taxonomy = load_place_chat_taxonomy()
    if taxonomy.version != settings.places_chat_taxonomy_version:
        raise RuntimeError(
            "PLACES_CHAT_TAXONOMY_VERSION does not match the bundled taxonomy"
        )
    return DeterministicPlaceChatIntentParser(taxonomy=taxonomy)


@lru_cache
def get_place_anchor_resolver() -> MainApiPlaceAnchorResolver | MockPlaceAnchorResolver:
    settings = get_settings()
    if settings.vector_store_provider == "mock":
        return MockPlaceAnchorResolver()
    return MainApiPlaceAnchorResolver(settings=settings)


@lru_cache
def get_hybrid_place_chat_retriever() -> HybridContentPlaceChatRetriever:
    settings = get_settings()
    return HybridContentPlaceChatRetriever(
        embedding_provider=get_embedding_provider(),
        place_repository=get_place_repository(),
        minimum_content_score=settings.places_chat_min_content_score,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        cache=get_place_chat_cache(),
    )


@lru_cache
def get_chat_place_recommendations_use_case() -> ChatPlaceRecommendationsUseCase:
    settings = get_settings()
    return ChatPlaceRecommendationsUseCase(
        intent_parser=get_place_chat_intent_parser(),
        anchor_resolver=get_place_anchor_resolver(),
        retriever=get_hybrid_place_chat_retriever(),
        llm_provider=get_llm_provider(),
        output_guard=PlaceChatOutputGuard(),
        ranking_version=settings.places_chat_ranking_version,
        taxonomy_version=settings.places_chat_taxonomy_version,
        llm_enabled=settings.places_chat_llm_enabled,
        anchor_ambiguity_delta=settings.places_chat_ambiguity_delta,
        minimum_intent_confidence=settings.places_chat_intent_min_confidence,
    )
