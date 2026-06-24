from typing import Any, Sequence

import pytest

from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceFilters
from app.modules.places.infrastructure.mock_place_repository import MockPlaceVectorRepository
from app.modules.places.infrastructure.simple_place_ranker import SimplePlaceRanker
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.llm.base import LLMProvider, LLMResult
from app.shared.nlp.llm.mock import MockLLMProvider
from app.shared.nlp.llm.output_guard import DEFAULT_PLACE_CHAT_FALLBACK, PlaceChatOutputGuard


@pytest.mark.asyncio
async def test_search_places_use_case_with_mock_providers() -> None:
    embedding_provider = MockEmbeddingProvider()
    use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=SimplePlaceRanker(),
    )

    result = await use_case.execute(
        query="lugares tranquilos para cenar",
        filters=PlaceFilters(city="Tuxtla Gutierrez", is_active=True),
        limit=3,
    )

    assert result.places
    assert all(place.city == "Tuxtla Gutierrez" for place in result.places)


@pytest.mark.asyncio
async def test_chat_places_returns_structured_places_with_llm_mock() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=SimplePlaceRanker(),
    )
    chat_use_case = ChatPlacesUseCase(
        search_use_case=search_use_case,
        llm_provider=MockLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
    )

    result = await chat_use_case.execute(
        message="quiero una cena tranquila",
        filters=PlaceFilters(city="Tuxtla Gutierrez", is_active=True),
        limit=3,
    )

    assert result.response_id.startswith("resp_")
    assert result.nlp_trace_id.startswith("trace_")
    assert result.message
    assert result.places
    assert result.metadata["used_llm"] is True


@pytest.mark.asyncio
async def test_chat_places_falls_back_when_llm_fails() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=SimplePlaceRanker(),
    )
    chat_use_case = ChatPlacesUseCase(
        search_use_case=search_use_case,
        llm_provider=FailingLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
    )

    result = await chat_use_case.execute(
        message="quiero salir con amigos",
        filters=PlaceFilters(city="Tuxtla Gutierrez", is_active=True),
        limit=3,
    )

    assert result.message == DEFAULT_PLACE_CHAT_FALLBACK
    assert result.places
    assert result.metadata["used_llm"] is False


class FailingLLMProvider(LLMProvider):
    provider_name = "failing"
    model_name = "failing-model"

    async def generate_place_chat_response(
        self,
        user_intent: str,
        region: str | None,
        places: Sequence[dict[str, Any]],
    ) -> LLMResult:
        raise RuntimeError("LLM failed")
