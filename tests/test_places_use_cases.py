from typing import Any, Sequence

import pytest

from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceFilters
from app.modules.places.infrastructure.bm25_place_ranker import Bm25PlaceRanker
from app.modules.places.infrastructure.mock_place_repository import MockPlaceVectorRepository
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.llm.base import LLMProvider, LLMResult, PlaceResponseMode
from app.shared.nlp.llm.mock import MockLLMProvider
from app.shared.nlp.llm.output_guard import DEFAULT_PLACE_CHAT_FALLBACK, PlaceChatOutputGuard


@pytest.mark.asyncio
async def test_search_places_use_case_with_mock_providers() -> None:
    embedding_provider = MockEmbeddingProvider()
    use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
    )

    result = await use_case.execute(
        query="lugares tranquilos para cenar",
        filters=PlaceFilters(city="Tuxtla Gutierrez", is_active=True),
        limit=3,
    )

    assert result.places
    assert all(place.city == "Tuxtla Gutierrez" for place in result.places)
    assert result.metrics.engine == "bm25"
    assert result.metrics.score_metric == "bm25"
    assert result.metrics.returned_count == len(result.places)


@pytest.mark.asyncio
async def test_search_places_ranks_with_bm25() -> None:
    embedding_provider = MockEmbeddingProvider()
    use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
    )

    result = await use_case.execute(
        query="atardecer fotos paseo",
        filters=PlaceFilters(is_active=True),
        limit=3,
    )

    assert result.places[0].id == "place_2"
    assert result.places[0].score > 0


@pytest.mark.asyncio
async def test_search_places_reports_zero_metrics_without_results() -> None:
    embedding_provider = MockEmbeddingProvider()
    use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
    )

    result = await use_case.execute(
        query="cafe",
        filters=PlaceFilters(city="Ciudad inexistente", is_active=True),
        limit=3,
    )

    assert result.places == []
    assert result.metrics.candidate_count == 0
    assert result.metrics.returned_count == 0
    assert result.metrics.min_score == 0.0
    assert result.metrics.max_score == 0.0
    assert result.metrics.mean_score == 0.0


@pytest.mark.asyncio
async def test_search_places_filters_candidates_with_nearby_ids() -> None:
    embedding_provider = MockEmbeddingProvider()
    nearby_provider = FakeNearbyPlaceProvider({"place_2", "place_6"})
    use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
        nearby_place_provider=nearby_provider,
    )

    result = await use_case.execute(
        query="quiero tomar fotos al atardecer",
        filters=PlaceFilters(is_active=True),
        limit=5,
        latitude=16.7531,
        longitude=-93.1156,
        radius_meters=10_000,
    )

    assert {place.id for place in result.places} <= {"place_2", "place_6"}
    assert nearby_provider.calls == 1
    assert result.metrics.location_filter_applied is True
    assert result.metrics.nearby_place_count == 2
    assert result.metrics.radius_meters == 10_000


@pytest.mark.asyncio
async def test_chat_places_returns_structured_places_with_llm_mock() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
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
async def test_recommend_places_calls_llm_and_returns_message() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
    )
    llm_provider = SpyLLMProvider()
    use_case = RecommendPlacesUseCase(
        search_use_case=search_use_case,
        llm_provider=llm_provider,
        output_guard=PlaceChatOutputGuard(),
    )

    result = await use_case.execute(
        query="quiero una cena tranquila",
        filters=PlaceFilters(city="Tuxtla Gutierrez", is_active=True),
        limit=3,
    )

    assert llm_provider.calls == 1
    assert result.message
    assert result.places
    assert result.metrics.engine == "bm25"
    assert result.metrics.returned_count == len(result.places)
    assert result.metrics.candidate_retrieval == "mock_embeddings"
    assert result.metrics.query_token_count > 0
    assert result.metrics.matched_query_token_count > 0
    assert result.metadata["used_llm"] is True
    assert result.metadata["ranking"] == "bm25"
    assert result.metadata["response_mode"] == "confident"
    assert llm_provider.response_modes == ["confident"]


@pytest.mark.asyncio
async def test_recommend_places_returns_empty_places_for_zero_bm25_score() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
    )
    use_case = RecommendPlacesUseCase(
        search_use_case=search_use_case,
        llm_provider=MockLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
    )

    result = await use_case.execute(
        query="xqzv blorf 998zz",
        filters=PlaceFilters(is_active=True),
        limit=3,
    )

    assert result.places == []
    assert result.metrics.max_score == 0.0
    assert result.metrics.returned_count == 0
    assert result.metrics.match_quality == "no_match"
    assert result.metadata["response_mode"] == "no_match"
    assert "no encontré lugares" in result.message.casefold()


@pytest.mark.asyncio
async def test_recommend_places_uses_cautious_message_below_threshold() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
        relevance_threshold=100.0,
    )
    use_case = RecommendPlacesUseCase(
        search_use_case=search_use_case,
        llm_provider=MockLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
    )

    result = await use_case.execute(
        query="cafe tranquilo",
        filters=PlaceFilters(is_active=True),
        limit=3,
    )

    assert 0 < result.metrics.max_score < result.metrics.relevance_threshold
    assert result.places
    assert result.metrics.match_quality == "low_confidence"
    assert result.metadata["response_mode"] == "low_confidence"
    assert "quizá" in result.message.casefold()


@pytest.mark.asyncio
async def test_chat_places_falls_back_when_llm_fails() -> None:
    embedding_provider = MockEmbeddingProvider()
    search_use_case = SearchPlacesUseCase(
        embedding_provider=embedding_provider,
        place_repository=MockPlaceVectorRepository(embedding_provider),
        ranker=Bm25PlaceRanker(),
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
        response_mode: PlaceResponseMode = "confident",
    ) -> LLMResult:
        raise RuntimeError("LLM failed")


class SpyLLMProvider(LLMProvider):
    provider_name = "spy-llama"
    model_name = "spy-model"

    def __init__(self) -> None:
        self.calls = 0
        self.response_modes: list[PlaceResponseMode] = []

    async def generate_place_chat_response(
        self,
        user_intent: str,
        region: str | None,
        places: Sequence[dict[str, Any]],
        response_mode: PlaceResponseMode = "confident",
    ) -> LLMResult:
        self.calls += 1
        self.response_modes.append(response_mode)
        place_names = ", ".join(str(place["name"]) for place in places[:2])
        return LLMResult(
            message=f"Estas opciones pueden encajar con tu plan: {place_names}.",
            provider=self.provider_name,
            model=self.model_name,
        )


class FakeNearbyPlaceProvider:
    def __init__(self, place_ids: set[str]) -> None:
        self._place_ids = place_ids
        self.calls = 0

    async def get_nearby_place_ids(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> set[str]:
        self.calls += 1
        return self._place_ids
