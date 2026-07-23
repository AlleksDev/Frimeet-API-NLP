from dataclasses import replace

import pytest

from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsUseCase,
)
from app.modules.places.domain.clarifications import (
    new_anchor_clarification,
    to_public_clarification,
)
from app.modules.places.domain.chat_intent import (
    ClarificationChoice,
    ConversationState,
    ExplicitTargetLocation,
    IntentAlternative,
    PendingClarification,
    PendingClarificationOption,
    PlaceChatCandidate,
    PlaceReference,
    ResolvedPlaceAnchor,
)
from app.modules.places.domain.models import PlaceCandidate
from app.modules.places.infrastructure.deterministic_intent_parser import (
    DeterministicPlaceChatIntentParser,
)
from app.modules.places.infrastructure.hybrid_chat_retriever import (
    HybridContentPlaceChatRetriever,
)
from app.modules.places.infrastructure.main_api_place_anchor_resolver import (
    MockPlaceAnchorResolver,
)
from app.modules.places.infrastructure.mock_place_repository import (
    MockPlaceVectorRepository,
)
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.llm.mock import MockLLMProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard


def build_use_case(
    *,
    llm_enabled: bool,
    anchor_resolver=None,
    intent_parser=None,
    retriever=None,
    nearby_place_provider=None,
):
    embedding = MockEmbeddingProvider(dimension=16)
    return ChatPlaceRecommendationsUseCase(
        intent_parser=intent_parser or DeterministicPlaceChatIntentParser(),
        anchor_resolver=anchor_resolver or MockPlaceAnchorResolver(),
        retriever=(
            retriever
            or HybridContentPlaceChatRetriever(
                embedding_provider=embedding,
                place_repository=MockPlaceVectorRepository(embedding),
                minimum_content_score=0.20,
            )
        ),
        llm_provider=MockLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
        ranking_version="places-chat-v2",
        taxonomy_version="places-taxonomy-v1",
        llm_enabled=llm_enabled,
        nearby_place_provider=nearby_place_provider,
    )


def test_public_anchor_buttons_are_bounded_for_the_main_api_contract() -> None:
    long_name = "Centro Historico " + ("muy grande " * 20)
    pending = new_anchor_clarification(
        "location_anchor",
        "centro",
        (
            ResolvedPlaceAnchor(place_id="anchor_1", name=long_name),
            ResolvedPlaceAnchor(place_id="anchor_2", name="Centro Dos"),
        ),
    )

    clarification = to_public_clarification(pending)

    assert len(clarification.options[0].label) <= 80
    assert len(clarification.options[0].message) <= 200


def test_open_category_buttons_use_safe_ids_and_preserve_raw_values() -> None:
    from app.modules.places.domain.clarifications import new_category_clarification

    pending = new_category_clarification(
        ("donas artesanales", "café de especialidad"),
        kind="intent_category",
    )
    clarification = to_public_clarification(pending)

    assert [option.option_id for option in pending.options] == [
        "donas_artesanales_1",
        "cafe_de_especialidad_2",
    ]
    assert [option.value for option in pending.options] == [
        "donas artesanales",
        "café de especialidad",
    ]
    assert [option.option_id for option in clarification.options] == [
        "donas_artesanales_1",
        "cafe_de_especialidad_2",
    ]


def test_category_buttons_humanize_labels_without_changing_ids() -> None:
    from app.modules.places.domain.clarifications import new_category_clarification

    pending = new_category_clarification(
        ("religious_organization", "ropa_barata"),
        kind="intent_category",
    )
    clarification = to_public_clarification(pending)

    assert [option.option_id for option in pending.options] == [
        "religious_organization",
        "ropa_barata",
    ]
    assert [option.value for option in pending.options] == [
        "religious_organization",
        "ropa_barata",
    ]
    assert [option.label for option in clarification.options] == [
        "Religious organization",
        "Ropa barata",
    ]
    assert [option.option_id for option in clarification.options] == [
        "religious_organization",
        "ropa_barata",
    ]


def test_known_category_buttons_use_spanish_fallback_labels() -> None:
    from app.modules.places.domain.clarifications import new_category_clarification

    clarification = to_public_clarification(
        new_category_clarification(
            ("sports", "park", "shopping"),
            kind="intent_category",
        )
    )

    assert [option.option_id for option in clarification.options] == [
        "sports",
        "park",
        "shopping",
    ]
    assert [option.label for option in clarification.options] == [
        "Deportes",
        "Parques",
        "Compras",
    ]


@pytest.mark.asyncio
async def test_llm_cannot_change_action_or_candidates() -> None:
    without_llm = await build_use_case(llm_enabled=False).execute(
        message="recomiendame una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )
    with_llm = await build_use_case(llm_enabled=True).execute(
        message="recomiendame una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert without_llm.action == with_llm.action == "recommendations"
    assert [item.place_id for item in without_llm.candidates] == [
        item.place_id for item in with_llm.candidates
    ]
    assert with_llm.used_llm is False
    assert with_llm.guard_reason == "candidate_name_deferred_to_main_api"


class LowConfidenceIntentParser:
    def __init__(self, alternatives: tuple[IntentAlternative, ...]) -> None:
        self._alternatives = alternatives

    def parse(
        self,
        message,
        state,
        has_user_location,
        clarification_choice=None,
    ):
        del message, clarification_choice
        parsed = DeterministicPlaceChatIntentParser().parse(
            message="una panaderia tranquila",
            state=state,
            has_user_location=has_user_location,
        )
        return replace(
            parsed,
            semantic_query="algo dulce tranquilo",
            confidence=0.45,
            alternatives=self._alternatives,
            category_source="semantic_activity",
        )


class CategoryEvidenceRetriever:
    def __init__(self, categories: tuple[str, ...] = ("bakery",)) -> None:
        self.calls = 0
        self.intents = []
        self._categories = categories

    async def retrieve(self, intent, limit):
        del limit
        self.calls += 1
        self.intents.append(intent)
        return [
            PlaceChatCandidate(
                place_id=f"{category}_1",
                name=f"Opcion local {category}",
                category=category,
                content_score=0.62 - (index * 0.01),
                semantic_score=0.64 - (index * 0.01),
                lexical_score=0.20,
                match_level="broad",
                matched_reasons=("algo dulce",),
                metadata={
                    "retrieval_diagnostics": {
                        "category_match": "exact",
                        "meets_minimum_content_score": True,
                    }
                },
            )
            for index, category in enumerate(self._categories)
        ]


@pytest.mark.asyncio
async def test_low_confidence_uses_dynamic_category_hypotheses() -> None:
    parser = LowConfidenceIntentParser(
        alternatives=(
            IntentAlternative(
                key="bakery",
                description="Panaderias artesanales",
                confidence=0.68,
            ),
            IntentAlternative(
                key="ice_cream",
                description="Postres y heladerias",
                confidence=0.64,
            ),
        )
    )

    result = await build_use_case(
        llm_enabled=False,
        intent_parser=parser,
        retriever=CategoryEvidenceRetriever(("bakery", "ice_cream")),
    ).execute(
        message="quiero algo dulce y tranquilo",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "clarification"
    assert result.clarification is not None
    assert [option.option_id for option in result.clarification.options] == [
        "bakery",
        "ice_cream",
    ]
    assert [option.label for option in result.clarification.options] == [
        "Panaderias artesanales",
        "Postres y heladerias",
    ]
    assert result.state_patch["target_category"] == "bakery"
    assert result.state_patch["soft_preferences"] == ["tranquilo"]
    assert [
        option["id"]
        for option in result.state_patch["pending_clarification"]["options"]
    ] == ["bakery", "ice_cream"]


@pytest.mark.asyncio
async def test_low_confidence_hypotheses_without_local_evidence_are_not_buttons() -> None:
    parser = LowConfidenceIntentParser(
        alternatives=(
            IntentAlternative(
                key="bakery",
                description="Panaderias artesanales",
                confidence=0.68,
                category_values=("bakery",),
            ),
            IntentAlternative(
                key="ice_cream",
                description="Postres y heladerias",
                confidence=0.64,
                category_values=("ice_cream",),
            ),
            IntentAlternative(
                key="office",
                description="Oficinas",
                confidence=0.63,
                category_values=("office",),
            ),
        )
    )

    result = await build_use_case(
        llm_enabled=False,
        intent_parser=parser,
        retriever=CategoryEvidenceRetriever(("bakery", "ice_cream")),
    ).execute(
        message="quiero algo dulce y tranquilo",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "clarification"
    assert result.clarification is not None
    assert [option.option_id for option in result.clarification.options] == [
        "bakery",
        "ice_cream",
    ]
    assert "office" not in {
        option["id"]
        for option in result.state_patch["pending_clarification"]["options"]
    }


@pytest.mark.asyncio
async def test_low_confidence_without_top_k_uses_retrieval_evidence_not_fixed_menu() -> None:
    retriever = CategoryEvidenceRetriever()
    result = await build_use_case(
        llm_enabled=False,
        intent_parser=LowConfidenceIntentParser(alternatives=()),
        retriever=retriever,
    ).execute(
        message="quiero algo dulce y tranquilo",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert retriever.calls == 1
    assert result.action == "recommendations"
    assert result.clarification is None
    assert [candidate.place_id for candidate in result.candidates] == ["bakery_1"]
    assert "intent_confidence" in result.unresolved
    assert "restaurante" not in result.message.casefold()
    assert "cafeteria" not in result.message.casefold()
    assert "parque" not in result.message.casefold()
    assert result.state_patch["soft_preferences"] == ["tranquilo"]


@pytest.mark.asyncio
async def test_candidate_categories_do_not_become_arbitrary_clarification_options() -> None:
    result = await build_use_case(
        llm_enabled=False,
        intent_parser=LowConfidenceIntentParser(alternatives=()),
        retriever=CategoryEvidenceRetriever(
            ("sports", "religious_organization", "office")
        ),
    ).execute(
        message="quiero ir a nadar",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert result.clarification is None
    assert {candidate.category for candidate in result.candidates} == {
        "sports",
        "religious_organization",
        "office",
    }


@pytest.mark.asyncio
async def test_supported_clarification_uses_localized_labels_without_counts() -> None:
    class LocalizedCategoryEvidenceRetriever(CategoryEvidenceRetriever):
        async def retrieve(self, intent, limit):
            candidates = await super().retrieve(intent, limit)
            labels = {
                "sports": "Deportes y centros acuaticos",
                "recreation": "Recreacion y balnearios",
            }
            return [
                replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        "category_label": labels[candidate.category],
                    },
                )
                for candidate in candidates
            ]

    parser = LowConfidenceIntentParser(
        alternatives=(
            IntentAlternative(
                key="sports",
                description="Sports",
                confidence=0.68,
                category_values=("sports",),
            ),
            IntentAlternative(
                key="recreation",
                description="Recreation",
                confidence=0.65,
                category_values=("recreation",),
            ),
        )
    )
    result = await build_use_case(
        llm_enabled=False,
        intent_parser=parser,
        retriever=LocalizedCategoryEvidenceRetriever(
            ("sports", "recreation")
        ),
    ).execute(
        message="quiero ir a nadar",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "clarification"
    assert result.clarification is not None
    labels = [option.label for option in result.clarification.options]
    assert labels == [
        "Deportes y centros acuaticos",
        "Recreacion y balnearios",
    ]
    assert all("opciones encontradas" not in label.casefold() for label in labels)
    assert all("coincidencias" not in label.casefold() for label in labels)


@pytest.mark.asyncio
async def test_low_quality_hypotheses_do_not_create_an_arbitrary_menu() -> None:
    parser = LowConfidenceIntentParser(
        alternatives=(
            IntentAlternative("bakery", "Panaderias", 0.55),
            IntentAlternative("outdoors", "Espacios abiertos", 0.52),
        )
    )

    result = await build_use_case(
        llm_enabled=False,
        intent_parser=parser,
        retriever=CategoryEvidenceRetriever(),
    ).execute(
        message="algo dificil de interpretar",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert result.clarification is None
    assert "intent_confidence" in result.unresolved


class WeakEvidenceRetriever:
    async def retrieve(self, intent, limit):
        del intent, limit
        return [
            PlaceChatCandidate(
                place_id="weak_1",
                name="Coincidencia tenue",
                category="cafe",
                content_score=0.0,
                semantic_score=0.0,
                lexical_score=0.0,
                match_level="broad",
                matched_reasons=(),
                metadata={
                    "retrieval_diagnostics": {
                        "meets_minimum_content_score": False,
                    }
                },
            )
        ]


class MixedEvidenceRetriever:
    async def retrieve(self, intent, limit):
        del intent, limit
        return [
            PlaceChatCandidate(
                place_id="baguette_shop",
                name="Lugar con baguettes",
                category="bakery",
                content_score=0.72,
                semantic_score=0.70,
                lexical_score=0.66,
                match_level="exact",
                matched_reasons=("baggets", "menu de baguettes"),
                metadata={
                    "menu_items": ["Baguette artesanal"],
                    "retrieval_diagnostics": {
                        "meets_minimum_content_score": True,
                    },
                },
            ),
            PlaceChatCandidate(
                place_id="tag_only_filler",
                name="Resultado respaldado solo por tag",
                category="restaurant",
                content_score=0.78,
                semantic_score=0.70,
                lexical_score=0.60,
                match_level="broad",
                matched_reasons=(),
                metadata={
                    "tags": ["baggets"],
                    "retrieval_diagnostics": {
                        "category_match": "compatible_tag_only",
                        "meets_minimum_content_score": True,
                    },
                },
            ),
        ]


@pytest.mark.asyncio
async def test_strong_results_are_not_padded_with_weak_candidates_and_explain_evidence() -> None:
    result = await build_use_case(
        llm_enabled=False,
        retriever=MixedEvidenceRetriever(),
    ).execute(
        message="quiero comer baggets",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert [candidate.place_id for candidate in result.candidates] == [
        "baguette_shop"
    ]
    assert "baggets" in result.message.casefold()
    assert "¡claro!" in result.message.casefold()
    assert "cards" not in result.message.casefold()


@pytest.mark.asyncio
async def test_explicit_category_does_not_return_weak_candidates() -> None:
    result = await build_use_case(
        llm_enabled=False,
        retriever=WeakEvidenceRetriever(),
    ).execute(
        message="una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.unresolved == ("retrieval_evidence",)
    assert result.candidates == ()
    assert "evidencia" in result.message.casefold()


class SequencedNearbyProvider:
    def __init__(self, responses: tuple[set[str], ...]) -> None:
        self.calls = []
        self._responses = responses

    async def get_nearby_place_ids(self, latitude, longitude, radius_meters):
        self.calls.append((latitude, longitude, radius_meters))
        index = len(self.calls) - 1
        if index >= len(self._responses):
            raise AssertionError("nearby provider received an unexpected call")
        return self._responses[index]


class ForbiddenRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, intent, limit):
        del intent, limit
        self.calls += 1
        raise AssertionError("retriever must not be called")


class ForbiddenNearbyProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def get_nearby_place_ids(self, latitude, longitude, radius_meters):
        del latitude, longitude, radius_meters
        self.calls += 1
        raise AssertionError("nearby provider must not be called")


class UnsupportedCategoryRetriever:
    def __init__(self) -> None:
        self.intents = []

    async def retrieve(self, intent, limit):
        del limit
        self.intents.append(intent)
        return [
            PlaceChatCandidate(
                place_id=f"park_{len(self.intents)}",
                name="Parque disponible",
                category="park",
                content_score=0.82,
                semantic_score=0.79,
                lexical_score=0.20,
                match_level="broad",
                matched_reasons=("aire libre",),
                metadata={
                    "retrieval_diagnostics": {
                        "category_match": "none",
                        "meets_minimum_content_score": True,
                    }
                },
            )
        ]


@pytest.mark.asyncio
async def test_greeting_does_not_call_retriever_or_nearby_provider() -> None:
    retriever = ForbiddenRetriever()
    nearby = ForbiddenNearbyProvider()

    result = await build_use_case(
        llm_enabled=False,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="ola",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.candidates == ()
    assert result.unresolved == ("non_search_input",)
    assert "hola" in result.message.casefold()
    assert retriever.calls == 0
    assert nearby.calls == 0


@pytest.mark.asyncio
async def test_empty_implicit_radius_expands_from_five_to_fifty_kilometers() -> None:
    nearby = SequencedNearbyProvider((set(), set()))
    retriever = ForbiddenRetriever()

    result = await build_use_case(
        llm_enabled=False,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.unresolved == ("nearby_catalog_empty",)
    assert nearby.calls == [
        (16.7531, -93.1156, 5_000),
        (16.7531, -93.1156, 50_000),
    ]
    assert result.location_directive.radius_meters == 50_000
    assert result.location_directive.strict_radius is False
    assert retriever.calls == 0


@pytest.mark.asyncio
async def test_empty_explicit_radius_does_not_expand() -> None:
    nearby = SequencedNearbyProvider((set(),))
    retriever = ForbiddenRetriever()

    result = await build_use_case(
        llm_enabled=False,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="una cafeteria a 2 km",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.unresolved == ("nearby_catalog_empty",)
    assert nearby.calls == [(16.7531, -93.1156, 2_000)]
    assert result.location_directive.radius_meters == 2_000
    assert result.location_directive.strict_radius is True
    assert retriever.calls == 0


@pytest.mark.asyncio
async def test_unsupported_category_expands_then_returns_category_availability() -> None:
    nearby = SequencedNearbyProvider(
        (
            {"park_near"},
            {"park_far", "park_near"},
        )
    )
    retriever = UnsupportedCategoryRetriever()

    result = await build_use_case(
        llm_enabled=False,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.candidates == ()
    assert result.unresolved == ("category_availability",)
    assert nearby.calls == [
        (16.7531, -93.1156, 5_000),
        (16.7531, -93.1156, 50_000),
    ]
    assert len(retriever.intents) == 2
    assert retriever.intents[0].hard_filters["place_ids"] == ("park_near",)
    assert retriever.intents[1].hard_filters["place_ids"] == (
        "park_far",
        "park_near",
    )
    assert result.location_directive.radius_meters == 50_000


class RecordingNearbyProvider:
    def __init__(self) -> None:
        self.call = None

    async def get_nearby_place_ids(self, latitude, longitude, radius_meters):
        self.call = (latitude, longitude, radius_meters)
        return {"near_2", "near_1"}


class RecordingFilteredRetriever:
    def __init__(self) -> None:
        self.intent = None

    async def retrieve(self, intent, limit):
        del limit
        self.intent = intent
        return [
            PlaceChatCandidate(
                place_id="near_1",
                name="Cercano",
                category="cafe",
                content_score=0.8,
                semantic_score=0.8,
                lexical_score=0.4,
                match_level="exact",
                matched_reasons=("cafe",),
                metadata={
                    "retrieval_diagnostics": {
                        "meets_minimum_content_score": True,
                    }
                },
            )
        ]


@pytest.mark.asyncio
async def test_current_location_ids_are_applied_before_content_retrieval() -> None:
    nearby = RecordingNearbyProvider()
    retriever = RecordingFilteredRetriever()

    result = await build_use_case(
        llm_enabled=False,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="una cafeteria",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert nearby.call == (16.7531, -93.1156, 5_000)
    assert retriever.intent.hard_filters["place_ids"] == ("near_1", "near_2")


@pytest.mark.asyncio
async def test_transient_geographic_place_ids_are_not_persisted_in_clarification() -> None:
    nearby = SequencedNearbyProvider(({"bakery_1", "ice_cream_1"},))
    retriever = CategoryEvidenceRetriever(("bakery", "ice_cream"))
    parser = LowConfidenceIntentParser(
        alternatives=(
            IntentAlternative(
                key="bakery",
                description="Panaderias artesanales",
                confidence=0.68,
                category_values=("bakery",),
            ),
            IntentAlternative(
                key="ice_cream",
                description="Postres y heladerias",
                confidence=0.64,
                category_values=("ice_cream",),
            ),
        )
    )

    result = await build_use_case(
        llm_enabled=False,
        intent_parser=parser,
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="quiero algo dulce y tranquilo",
        state=ConversationState(hard_filters={"city": "Tuxtla"}),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "clarification"
    assert retriever.intents[0].hard_filters["place_ids"] == (
        "bakery_1",
        "ice_cream_1",
    )
    assert result.state_patch["hard_filters"] == {"city": "Tuxtla"}
    assert "place_ids" not in result.state_patch["hard_filters"]
    assert result.location_directive.radius_meters == 5_000


class CoordinateAnchorResolver:
    async def resolve(self, text, city, state, limit=3):
        del text, city, state, limit
        return [
            ResolvedPlaceAnchor(
                place_id="anchor_centro",
                name="Centro",
                latitude=16.75,
                longitude=-93.12,
                score=0.95,
            )
        ]


@pytest.mark.asyncio
async def test_explicit_anchor_coordinates_filter_before_retrieval() -> None:
    nearby = RecordingNearbyProvider()
    retriever = RecordingFilteredRetriever()

    result = await build_use_case(
        llm_enabled=False,
        anchor_resolver=CoordinateAnchorResolver(),
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="una cafeteria cerca del centro",
        state=ConversationState(),
        user_latitude=16.70,
        user_longitude=-93.10,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert result.location_directive.source == "explicit_anchor"
    assert nearby.call == (16.75, -93.12, 5_000)
    assert retriever.intent.hard_filters["place_ids"] == ("near_1", "near_2")


@pytest.mark.asyncio
async def test_persisted_resolved_anchor_keeps_nearby_filter_on_followup() -> None:
    nearby = RecordingNearbyProvider()
    retriever = RecordingFilteredRetriever()

    result = await build_use_case(
        llm_enabled=False,
        anchor_resolver=CoordinateAnchorResolver(),
        retriever=retriever,
        nearby_place_provider=nearby,
    ).execute(
        message="otra cafeteria tranquila",
        state=ConversationState(
            explicit_target_location=ExplicitTargetLocation(
                anchor_text="Centro",
                place_id="anchor_centro",
                label="Centro",
                radius_meters=3_000,
                strict_radius=True,
            )
        ),
        user_latitude=None,
        user_longitude=None,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert result.location_directive.source == "state_anchor"
    assert nearby.call == (16.75, -93.12, 3_000)
    assert retriever.intent.hard_filters["place_ids"] == ("near_1", "near_2")


class AmbiguousAnchorResolver:
    async def resolve(self, text, city, state, limit=3):
        del text, city, state, limit
        return [
            ResolvedPlaceAnchor(
                place_id="anchor_1",
                name="Centro Uno",
                score=0.90,
            ),
            ResolvedPlaceAnchor(
                place_id="anchor_2",
                name="Centro Dos",
                score=0.82,
            ),
        ]


@pytest.mark.asyncio
async def test_ambiguous_anchor_stops_before_retrieval() -> None:
    result = await build_use_case(
        llm_enabled=False,
        anchor_resolver=AmbiguousAnchorResolver(),
    ).execute(
        message="recomiendame una cafeteria cerca del centro",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "clarification"
    assert result.candidates == ()
    assert result.unresolved == ("location_anchor",)
    assert result.clarification is not None
    assert [option.label for option in result.clarification.options] == [
        "Centro Uno",
        "Centro Dos",
    ]


@pytest.mark.asyncio
async def test_ambiguous_anchor_button_resolves_exactly_once() -> None:
    use_case = build_use_case(
        llm_enabled=False,
        anchor_resolver=AmbiguousAnchorResolver(),
    )
    first = await use_case.execute(
        message="recomiendame una cafeteria cerca del centro",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )
    payload = first.state_patch["pending_clarification"]
    pending = PendingClarification(
        clarification_id=payload["id"],
        kind=payload["kind"],
        location_anchor_text=payload["location_anchor_text"],
        options=tuple(
            PendingClarificationOption(
                option_id=option["id"],
                value=option["value"],
                label=option["label"],
                place_id=option["place_id"],
                attributes=tuple(option["attributes"]),
            )
            for option in payload["options"]
        ),
    )

    second = await use_case.execute(
        message="Centro Dos",
        state=ConversationState(
            target_category=first.state_patch["target_category"],
            pending_clarification=pending,
        ),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
        clarification_choice=ClarificationChoice(
            clarification_id=pending.clarification_id,
            option_id="option_2",
        ),
    )

    assert second.action == "recommendations"
    assert second.clarification is None
    assert second.state_patch["pending_clarification"] is None
    assert second.location_directive.anchor_place_id == "anchor_2"


class MissingAnchorResolver:
    async def resolve(self, text, city, state, limit=3):
        del text, city, state, limit
        return []


@pytest.mark.asyncio
async def test_missing_required_anchor_stops_before_retrieval() -> None:
    result = await build_use_case(
        llm_enabled=False,
        anchor_resolver=MissingAnchorResolver(),
    ).execute(
        message="recomiendame una cafeteria cerca de una zona desconocida",
        state=ConversationState(),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "no_match"
    assert result.candidates == ()
    assert result.unresolved == ()


class UnfilteredThemeRepository:
    source_name = "test"

    async def search(self, embedding, filters, limit):
        del embedding, filters, limit
        return [
            PlaceCandidate(
                id="park",
                name="Parque con Hello Kitty",
                category="park",
                score=1.0,
                metadata={"tags": "hello kitty caricaturas"},
            ),
            PlaceCandidate(
                id="broad",
                name="Cafe Generico",
                category="cafe",
                score=0.99,
                metadata={"tags": "cafe postres"},
            ),
            PlaceCandidate(
                id="family",
                name="Cafe de Caricaturas",
                category="cafe",
                score=0.90,
                metadata={"tags": "caricaturas personajes"},
            ),
            PlaceCandidate(
                id="exact",
                name="Cafe Kitty",
                category="cafe",
                score=0.20,
                metadata={"tags": "hello kitty"},
            ),
        ]


class MixedCategoryRepository:
    source_name = "test"

    async def search(self, embedding, filters, limit):
        del embedding, filters, limit
        return [
            PlaceCandidate(
                id="park_exact",
                name="Parque Central",
                category="park",
                score=0.0,
                metadata={"tags": "parque aire libre"},
            ),
            PlaceCandidate(
                id="cinema_compatible",
                name="Cinepolis Centro",
                category="entertainment",
                score=0.0,
                metadata={"tags": "Cine peliculas estrenos"},
            ),
            PlaceCandidate(
                id="entertainment_noise",
                name="Boliche Centro",
                category="entertainment",
                score=1.0,
                metadata={"tags": "boliche juegos diversion"},
                document="entertainment entretenimiento diversion cine actividades",
            ),
            PlaceCandidate(
                id="shopping_exact",
                name="Plaza Central",
                category="shopping_mall",
                score=0.0,
                metadata={"tags": "compras tiendas ropa"},
            ),
            PlaceCandidate(
                id="family_noise",
                name="Salon de fiestas",
                category="family",
                score=1.0,
                metadata={"tags": "fiestas infantiles salon"},
            ),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_id"),
    (
        ("parque", "park_exact"),
        ("quiero ir al parque", "park_exact"),
        ("cine", "cinema_compatible"),
        ("quiero ir al cine", "cinema_compatible"),
        ("compras", "shopping_exact"),
        ("shopping", "shopping_exact"),
    ),
)
async def test_short_category_queries_survive_sparse_scores_without_being_filtered(
    message: str,
    expected_id: str,
) -> None:
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=MockEmbeddingProvider(dimension=16),
        place_repository=MixedCategoryRepository(),
        minimum_content_score=0.95,
    )
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    candidates = await retriever.retrieve(intent=intent, limit=5)

    candidate_ids = [candidate.place_id for candidate in candidates]
    assert expected_id in candidate_ids
    assert set(candidate_ids) == {
        "park_exact",
        "cinema_compatible",
        "entertainment_noise",
        "shopping_exact",
        "family_noise",
    }
    assert all(
        candidate.metadata["retrieval_diagnostics"]["content_quality"] == "weak"
        for candidate in candidates
    )


@pytest.mark.asyncio
async def test_category_is_ranking_evidence_instead_of_a_hard_filter() -> None:
    embedding = MockEmbeddingProvider(dimension=16)
    retriever = HybridContentPlaceChatRetriever(
        embedding_provider=embedding,
        place_repository=UnfilteredThemeRepository(),
        minimum_content_score=0.0,
    )
    intent = DeterministicPlaceChatIntentParser().parse(
        message="cafeterias de Hello Kitty",
        state=ConversationState(),
        has_user_location=True,
    )

    candidates = await retriever.retrieve(intent=intent, limit=10)

    assert {candidate.place_id for candidate in candidates} == {
        "park",
        "exact",
        "family",
        "broad",
    }
    diagnostics = {
        candidate.place_id: candidate.metadata["retrieval_diagnostics"]
        for candidate in candidates
    }
    assert diagnostics["park"]["category_match"] == "none"
    assert diagnostics["exact"]["category_match"] == "exact"


class ReferenceLocationResolver:
    async def resolve(self, text, city, state, limit=3):
        del city, state, limit
        if text == "parque central":
            return [
                ResolvedPlaceAnchor(
                    place_id="park_anchor",
                    name="Parque Central",
                    latitude=16.7531,
                    longitude=-93.1156,
                    score=0.95,
                )
            ]
        return [
            ResolvedPlaceAnchor(
                place_id="kitty_far",
                name="Hello Kitty Cafe Norte",
                latitude=16.90,
                longitude=-93.20,
                attributes=("hello", "kitty", "caricaturas"),
                score=0.90,
            ),
            ResolvedPlaceAnchor(
                place_id="kitty_near",
                name="Hello Kitty Cafe Centro",
                latitude=16.7540,
                longitude=-93.1160,
                attributes=("hello", "kitty", "tematica"),
                score=0.90,
            ),
        ]


@pytest.mark.asyncio
async def test_reference_location_hint_disambiguates_without_moving_results() -> None:
    result = await build_use_case(
        llm_enabled=False,
        anchor_resolver=ReferenceLocationResolver(),
    ).execute(
        message="quiero cafeterias parecidas",
        state=ConversationState(
            target_category="cafe",
            soft_preferences=("hello_kitty", "tematica", "caricaturas"),
            reference=PlaceReference(
                entity="hello kitty",
                attributes=("hello_kitty", "tematica", "caricaturas"),
                location_hint_text="parque central",
            ),
        ),
        user_latitude=16.7531,
        user_longitude=-93.1156,
        candidate_limit=5,
        result_limit=3,
    )

    assert result.action == "recommendations"
    assert result.state_patch["reference"]["place_id"] == "kitty_near"
    assert result.state_patch["reference"]["location_hint_text"] == "parque central"
    assert result.location_directive.source == "user_current"
