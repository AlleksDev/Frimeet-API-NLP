import pytest

from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsUseCase,
)
from app.modules.places.domain.chat_intent import (
    ConversationState,
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


def build_use_case(*, llm_enabled: bool, anchor_resolver=None):
    embedding = MockEmbeddingProvider(dimension=16)
    return ChatPlaceRecommendationsUseCase(
        intent_parser=DeterministicPlaceChatIntentParser(),
        anchor_resolver=anchor_resolver or MockPlaceAnchorResolver(),
        retriever=HybridContentPlaceChatRetriever(
            embedding_provider=embedding,
            place_repository=MockPlaceVectorRepository(embedding),
            minimum_content_score=0.20,
        ),
        llm_provider=MockLLMProvider(),
        output_guard=PlaceChatOutputGuard(),
        ranking_version="places-chat-v2",
        taxonomy_version="places-taxonomy-v1",
        llm_enabled=llm_enabled,
    )


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

    assert result.action == "clarification"
    assert result.candidates == ()
    assert result.unresolved == ("location_anchor",)


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


@pytest.mark.asyncio
async def test_hard_category_and_controlled_theme_relaxation() -> None:
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

    assert [candidate.place_id for candidate in candidates] == [
        "exact",
        "family",
        "broad",
    ]
    assert [candidate.match_level for candidate in candidates] == [
        "exact",
        "family",
        "broad",
    ]


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
