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
    soft_preferences: tuple[str, ...] = (),
    exclusions: tuple[str, ...] = (),
) -> ParsedPlaceChatIntent:
    return ParsedPlaceChatIntent(
        action="recommendations",
        target_category=target_category,
        category_values=category_values,
        hard_filters=hard_filters or {},
        soft_preferences=soft_preferences,
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
        "attribute_conflict_affinity": 0.0,
        "attribute_conflicts": [],
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


@pytest.mark.asyncio
async def test_positive_attributes_rank_false_is_penalized_and_unknown_is_neutral() -> None:
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="confirmed",
                name="Centro Acuatico",
                category="sports_center",
                score=0.0,
                metadata={
                    "attribute_terms": [
                        "estacionamiento",
                        "pantallas",
                        "apto para refrescarse nadar actividades acuaticas",
                    ],
                    "attribute_states": {
                        "has_parking": True,
                        "has_screens": True,
                        "good_for_cooling_off": "yes",
                    },
                    "source_fields_present": ["name", "category", "facets"],
                },
            ),
            PlaceCandidate(
                id="explicit_false",
                name="Cancha Techada",
                category="sports_center",
                score=0.0,
                metadata={
                    "attribute_states": {
                        "has_parking": False,
                        "has_screens": False,
                        "good_for_cooling_off": "no",
                    },
                    "negative_attribute_terms": [
                        "estacionamiento",
                        "pantallas",
                        "good for cooling off",
                    ],
                    "source_fields_present": ["name", "category"],
                },
            ),
            PlaceCandidate(
                id="unknown_sparse",
                name="Unidad Deportiva",
                category="sports_center",
                score=0.0,
                metadata={
                    # Missing description and NULL attributes are unknown, not
                    # evidence that the place lacks the requested facilities.
                    "source_fields_present": ["name", "category"],
                },
            ),
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
        minimum_content_score=0.20,
    )

    candidates = await retriever.retrieve(
        intent=_intent(
            "estacionamiento pantallas para refrescarse",
            soft_preferences=(
                "estacionamiento",
                "pantallas",
                "para_refrescarse",
            ),
        ),
        limit=5,
    )

    by_id = {candidate.place_id: candidate for candidate in candidates}
    assert set(by_id) == {"confirmed", "explicit_false", "unknown_sparse"}
    assert candidates[0].place_id == "confirmed"
    assert [candidate.place_id for candidate in candidates] == [
        "confirmed",
        "unknown_sparse",
        "explicit_false",
    ]
    assert by_id["confirmed"].content_score > by_id["explicit_false"].content_score
    assert by_id["confirmed"].content_score > by_id["unknown_sparse"].content_score
    assert set(by_id["confirmed"].matched_reasons) == {
        "estacionamiento",
        "pantallas",
        "para_refrescarse",
    }
    assert by_id["explicit_false"].matched_reasons == ()
    assert by_id["unknown_sparse"].matched_reasons == ()

    confirmed_diagnostics = by_id["confirmed"].metadata["retrieval_diagnostics"]
    false_diagnostics = by_id["explicit_false"].metadata[
        "retrieval_diagnostics"
    ]
    unknown_diagnostics = by_id["unknown_sparse"].metadata[
        "retrieval_diagnostics"
    ]
    assert confirmed_diagnostics["meets_minimum_content_score"] is True
    assert false_diagnostics["attribute_conflict_affinity"] == 1.0
    assert set(false_diagnostics["attribute_conflicts"]) == {
        "estacionamiento",
        "pantallas",
        "para_refrescarse",
    }
    assert unknown_diagnostics["attribute_conflict_affinity"] == 0.0
    assert unknown_diagnostics["attribute_conflicts"] == []
    assert unknown_diagnostics["minimum_content_score"] == pytest.approx(0.13)
    assert unknown_diagnostics["configured_minimum_content_score"] == 0.20
    assert unknown_diagnostics["meets_minimum_content_score"] is False


@pytest.mark.asyncio
async def test_all_structured_positive_attributes_match_without_lexical_labels() -> None:
    preferences = (
        "area_infantil",
        "banos",
        "estacionamiento",
        "para_refrescarse",
        "economico",
        "entretenimiento",
        "romantico",
        "familiar",
        "para_amigos",
        "pantallas",
        "asientos",
        "comida_exterior",
        "mochila",
    )
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="structured",
                name="Lugar Uno",
                category="other",
                score=0.0,
                metadata={
                    "attribute_states": {
                        "has_kids_playroom": True,
                        "kid_friendly": "yes",
                        "has_restrooms": True,
                        "has_parking": True,
                        "good_for_cooling_off": "yes",
                        "price_tier": "cheap",
                        "has_entertainment": True,
                        "has_romantic_space": True,
                        "has_family_space": True,
                        "has_friends_space": True,
                        "has_screens": True,
                        "has_seating": True,
                        "allows_outside_food": True,
                        "allows_backpack": True,
                    }
                },
            )
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
    )

    candidates = await retriever.retrieve(
        intent=_intent(
            "atributos confirmados",
            soft_preferences=preferences,
        ),
        limit=3,
    )

    assert set(candidates[0].matched_reasons) == set(preferences)
    assert candidates[0].match_level == "family"


@pytest.mark.asyncio
async def test_generic_entertainment_does_not_claim_music() -> None:
    repository = RecordingRepository(
        [
            PlaceCandidate(
                id="generic-entertainment",
                name="Lugar Uno",
                category="other",
                score=0.0,
                metadata={"attribute_states": {"has_entertainment": True}},
            )
        ]
    )
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=repository,
    )

    candidates = await retriever.retrieve(
        intent=_intent("musica", soft_preferences=("musica",)),
        limit=3,
    )

    assert candidates[0].matched_reasons == ()
