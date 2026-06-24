from functools import lru_cache

from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.infrastructure.mock_place_repository import MockPlaceVectorRepository
from app.modules.places.infrastructure.simple_place_ranker import SimplePlaceRanker
from app.shared.cache.memory import SimpleTTLCache
from app.shared.dependencies import get_embedding_provider, get_llm_provider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard


@lru_cache
def get_place_repository() -> MockPlaceVectorRepository:
    return MockPlaceVectorRepository(embedding_provider=get_embedding_provider())


@lru_cache
def get_place_ranker() -> SimplePlaceRanker:
    return SimplePlaceRanker()


@lru_cache
def get_place_search_cache() -> SimpleTTLCache:
    return SimpleTTLCache(default_ttl_seconds=60)


@lru_cache
def get_search_places_use_case() -> SearchPlacesUseCase:
    return SearchPlacesUseCase(
        embedding_provider=get_embedding_provider(),
        place_repository=get_place_repository(),
        ranker=get_place_ranker(),
        cache=get_place_search_cache(),
    )


@lru_cache
def get_recommend_places_use_case() -> RecommendPlacesUseCase:
    return RecommendPlacesUseCase(search_use_case=get_search_places_use_case())


@lru_cache
def get_chat_places_use_case() -> ChatPlacesUseCase:
    return ChatPlacesUseCase(
        search_use_case=get_search_places_use_case(),
        llm_provider=get_llm_provider(),
        output_guard=PlaceChatOutputGuard(),
    )
