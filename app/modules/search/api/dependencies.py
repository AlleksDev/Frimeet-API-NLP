from functools import lru_cache

from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.application.use_cases.search_all import SearchAllUseCase
from app.modules.search.domain.models import ALL_SEARCH_RESOURCE_TYPES, SearchResourceType
from app.modules.search.infrastructure.mock_provider import (
    MockHybridSearchProvider,
    get_mock_search_documents,
)
from app.modules.search.infrastructure.pgvector_provider import (
    PgvectorHybridSearchProvider,
    PgvectorPlaceSearchProvider,
)
from app.modules.places.infrastructure.main_api_nearby_place_provider import (
    MainApiNearbyPlaceProvider,
)
from app.shared.config.settings import get_settings
from app.shared.dependencies import get_embedding_provider
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient


@lru_cache
def get_global_search_providers() -> tuple[SearchProvider, ...]:
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    if settings.vector_store_provider == "aws_pgvector":
        vector_client = AwsPgvectorClient(settings, role="reader")
        return tuple(
            (
                PgvectorPlaceSearchProvider(vector_client)
                if resource_type == SearchResourceType.PLACES
                else PgvectorHybridSearchProvider(resource_type, vector_client)
            )
            for resource_type in ALL_SEARCH_RESOURCE_TYPES
        )

    documents = get_mock_search_documents()
    return tuple(
        MockHybridSearchProvider(
            resource_type=resource_type,
            embedding_provider=embedding_provider,
            documents=documents[resource_type],
        )
        for resource_type in ALL_SEARCH_RESOURCE_TYPES
    )


@lru_cache
def get_search_all_use_case() -> SearchAllUseCase:
    return SearchAllUseCase(
        embedding_provider=get_embedding_provider(),
        providers=list(get_global_search_providers()),
        nearby_place_provider=MainApiNearbyPlaceProvider(get_settings()),
    )
