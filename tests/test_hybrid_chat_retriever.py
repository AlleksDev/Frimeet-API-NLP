import pytest

from app.modules.places.domain.chat_intent import (
    ConversationStatePatch,
    LocationIntent,
    ParsedPlaceChatIntent,
)
from app.modules.places.domain.models import PlaceCandidate
from app.modules.places.infrastructure.hybrid_chat_retriever import (
    HybridContentPlaceChatRetriever,
)
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider


def _intent(
    query: str,
    *,
    target_category: str | None = None,
    category_values: tuple[str, ...] = (),
    hard_filters: dict[str, object] | None = None,
    exclusions: tuple[str, ...] = (),
) -> ParsedPlaceChatIntent:
    return ParsedPlaceChatIntent(
        action="recommendations",
        target_category=target_category,
        category_values=category_values,
        hard_filters=hard_filters or {},
        soft_preferences=(),
        exclusions=exclusions,
        reference=None,
        location=LocationIntent(
            scope="target_results",
            source="user_current",
        ),
        semantic_query=query,
        confidence=0.90,
        state_patch=ConversationStatePatch(),
    )


class RecordingRepository:
    source_name = "test"

    def __init__(self, candidates: list[PlaceCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[dict[str, object]] = []

    async def search(self, embedding, filters, limit):
        self.calls.append(
            {
                "embedding": embedding,
                "filters": filters,
                "limit": limit,
            }
        )
        return self.candidates


@pytest.mark.asyncio
async def test_open_vocabulary_recommendation_retrieves_without_category() -> None:
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="donut_shop",
                name="Dulce Circular",
                category="bakery",
                score=0.0,
                metadata={"tags": "postres artesanales"},
            )
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
        minimum_content_score=0.95,
    )

    candidates = await retriever.retrieve(
        intent=_intent(
            "donas",
            hard_filters={"city": "Tuxtla Gutierrez", "state": "Chiapas"},
        ),
        limit=3,
    )

    assert [candidate.place_id for candidate in candidates] == ["donut_shop"]
    assert repository.calls[0]["filters"].categories is None
    assert repository.calls[0]["filters"].city == "Tuxtla Gutierrez"
    assert repository.calls[0]["filters"].state == "Chiapas"
    diagnostics = candidates[0].metadata["retrieval_diagnostics"]
    assert diagnostics == {
        "category_affinity": 0.0,
        "category_match": "not_requested",
        "exclusion_affinity": 0.0,
        "exclusion_matches": [],
        "content_quality": "weak",
        "meets_minimum_content_score": False,
        "minimum_content_score": 0.95,
        "query_token_count": 1,
    }


@pytest.mark.asyncio
async def test_category_boosts_ranking_but_does_not_remove_other_categories() -> None:
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="park",
                name="Parque Vecinal",
                category="park",
                score=0.2,
            ),
            PlaceCandidate(
                id="cafe",
                name="Salon Urbano",
                category="cafe",
                score=0.2,
            ),
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
    )

    candidates = await retriever.retrieve(
        intent=_intent(
            "sitio agradable",
            target_category="cafe",
            category_values=("cafe", "cafeteria"),
        ),
        limit=5,
    )

    assert [candidate.place_id for candidate in candidates] == ["cafe", "park"]
    assert candidates[0].metadata["retrieval_diagnostics"]["category_match"] == (
        "exact"
    )
    assert candidates[1].metadata["retrieval_diagnostics"]["category_match"] == (
        "none"
    )


class HybridRepository:
    source_name = "test_hybrid"

    def __init__(self) -> None:
        self.hybrid_call: dict[str, object] | None = None

    async def search_hybrid(self, query_text, embedding, filters, limit):
        self.hybrid_call = {
            "query_text": query_text,
            "embedding": embedding,
            "filters": filters,
            "limit": limit,
        }
        return [
            PlaceCandidate(
                id="hybrid_result",
                name="Resultado Hibrido",
                category="bakery",
                # The fused repository score is not a semantic score.  This
                # row came from the independent lexical candidate pool.
                score=0.99,
                metadata={"lexical_score": 1.0},
            )
        ]

    async def search(self, embedding, filters, limit):
        del embedding, filters, limit
        raise AssertionError("search must not run when search_hybrid is available")


@pytest.mark.asyncio
async def test_optional_repository_hybrid_search_is_preferred() -> None:
    repository = HybridRepository()
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
    )

    candidates = await retriever.retrieve(
        intent=_intent("Donas cerca"),
        limit=3,
    )

    assert [candidate.place_id for candidate in candidates] == ["hybrid_result"]
    assert repository.hybrid_call is not None
    assert repository.hybrid_call["query_text"] == "Donas cerca"
    assert repository.hybrid_call["filters"].categories is None
    assert repository.hybrid_call["limit"] == 40
    assert candidates[0].semantic_score == 0.0
    assert candidates[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_exclusions_penalize_instead_of_dropping_and_respect_negation() -> None:
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="quiet",
                name="Patio Sereno",
                category="cafe",
                score=0.5,
                document="Un espacio tranquilo sin ruido exterior",
            ),
            PlaceCandidate(
                id="noisy",
                name="Foro Central",
                category="cafe",
                score=0.5,
                document="Musica y ruido durante toda la noche",
            ),
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
    )

    candidates = await retriever.retrieve(
        intent=_intent("lugar tranquilo", exclusions=("ruido",)),
        limit=5,
    )

    assert [candidate.place_id for candidate in candidates] == ["quiet", "noisy"]
    quiet_diagnostics = candidates[0].metadata["retrieval_diagnostics"]
    noisy_diagnostics = candidates[1].metadata["retrieval_diagnostics"]
    assert quiet_diagnostics["exclusion_affinity"] == 0.0
    assert noisy_diagnostics["exclusion_affinity"] == 1.0
    assert noisy_diagnostics["exclusion_matches"] == ["ruido"]
    assert candidates[0].content_score > candidates[1].content_score
