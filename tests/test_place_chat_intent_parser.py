import pytest

from app.modules.places.domain.chat_intent import (
    ClarificationChoice,
    ConversationState,
    ExplicitTargetLocation,
    PendingClarification,
    PlaceCategoryInference,
    PlaceReference,
)
from app.modules.places.domain.errors import ClarificationStateMismatchError
from app.modules.places.infrastructure.deterministic_intent_parser import (
    DeterministicPlaceChatIntentParser,
)


def test_category_is_not_polluted_by_the_location_anchor() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="Recomiendame alguna cafeteria cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert intent.category_values == (
        "cafe",
        "café",
        "cafeteria",
        "coffee_shop",
        "coffee shop",
    )
    assert intent.semantic_query == "cafe cafeteria"
    assert intent.location.scope == "target_results"
    assert intent.location.anchor_text == "parque central"


def test_reference_and_location_scope_ambiguity_requires_clarification() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="cafeterias como la de Hello Kitty cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "clarification"
    assert intent.unresolved == ("location_scope",)
    assert [alternative.key for alternative in intent.alternatives] == [
        "target_results",
        "reference_entity",
    ]
    assert intent.semantic_query == ""
    assert intent.state_patch.target_category == "cafe"
    assert intent.state_patch.reference is not None
    pending = intent.state_patch.pending_clarification
    assert pending is not None
    assert pending.kind == "location_scope"
    assert pending.location_anchor_text == "parque central"
    assert pending.clarification_id
    assert [option.option_id for option in pending.options] == [
        "target_results",
        "reference_entity",
    ]
    assert intent.clarification is not None
    assert intent.clarification.clarification_id == pending.clarification_id


def test_continuation_inherits_category_and_adds_price_filter() -> None:
    state = ConversationState(
        target_category="cafe",
        soft_preferences=("tematica",),
        reference=PlaceReference(entity="hello kitty"),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="mas barato y no tan lejos",
        state=state,
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert intent.hard_filters["price_preference"] == "lower"
    assert intent.reference == state.reference
    assert intent.location.source == "user_current"
    assert intent.state_patch.as_dict()["hard_filters"] == {
        "price_preference": "lower"
    }


def test_new_category_clears_incompatible_reference_and_preferences() -> None:
    state = ConversationState(
        target_category="cafe",
        soft_preferences=("hello_kitty", "tematica"),
        reference=PlaceReference(entity="hello kitty"),
        explicit_target_location=ExplicitTargetLocation(anchor_text="parque central"),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="mejor recomiendame un parque",
        state=state,
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "park"
    assert intent.reference is None
    assert intent.soft_preferences == ()
    assert intent.state_patch.clear_reference is True
    assert intent.location.source == "conversation_state"


def test_explicit_radius_is_strict_and_removed_from_anchor_text() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="una cafeteria cerca del Parque Central a 2 km",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.location.anchor_text == "parque central"
    assert intent.location.radius_meters == 2000
    assert intent.location.strict_radius is True


def test_missing_category_requires_clarification() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero un lugar bonito para salir",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "clarification"
    assert intent.unresolved == ("target_category",)


def test_food_activity_defaults_to_restaurant_without_clarification() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero comer algo",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "restaurant"
    assert intent.category_values == ("restaurant", "restaurante", "comedor")
    assert intent.semantic_query == "restaurant restaurante comer"
    assert intent.state_patch.target_category == "restaurant"
    assert intent.confidence == 0.88


def test_explicit_category_wins_over_an_activity_default() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero comer algo en una cafeteria",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert intent.category_values == (
        "cafe",
        "café",
        "cafeteria",
        "coffee_shop",
        "coffee shop",
    )
    assert intent.state_patch.target_category == "cafe"


def test_other_high_confidence_activities_use_helpful_defaults() -> None:
    parser = DeterministicPlaceChatIntentParser()
    examples = {
        "quiero hacer ejercicio": "sports",
        "quiero ver una pelicula": "cinema",
        "quiero comprar algo": "shopping",
        "necesito un lugar donde dormir": "lodging",
    }

    for message, expected_category in examples.items():
        intent = parser.parse(
            message=message,
            state=ConversationState(),
            has_user_location=True,
        )

        assert intent.action == "recommendations"
        assert intent.target_category == expected_category


@pytest.mark.parametrize(
    ("message", "expected_category", "expected_query_term"),
    (
        ("parque", "park", "parque"),
        ("quiero ir al parque", "park", "parque"),
        ("cine", "cinema", "cine"),
        ("quiero ir al cine", "cinema", "cine"),
        ("compras", "shopping", "compras"),
        ("shopping", "shopping", "shopping"),
        ("quiero ir de compras", "shopping", "compras"),
    ),
)
def test_short_category_requests_keep_local_retrieval_vocabulary(
    message: str,
    expected_category: str,
    expected_query_term: str,
) -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == expected_category
    assert expected_query_term in intent.semantic_query.split()


def test_semantic_activity_inference_handles_non_literal_food_request() -> None:
    class FoodActivityClassifier:
        def classify(self, text: str) -> PlaceCategoryInference | None:
            if "apetecen" in text and "tacos" in text:
                return PlaceCategoryInference(
                    category="restaurant",
                    confidence=0.82,
                    source="semantic_activity",
                )
            return None

    intent = DeterministicPlaceChatIntentParser(
        activity_classifier=FoodActivityClassifier(),
    ).parse(
        message="Después de todo el día se me apetecen unos buenos tacos",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "restaurant"
    assert intent.confidence == 0.82


def test_new_clear_intent_cancels_a_stale_pending_clarification() -> None:
    state = ConversationState(
        target_category="cafe",
        soft_preferences=("tematica",),
        reference=PlaceReference(entity="hello kitty"),
        pending_clarification=PendingClarification(
            kind="location_scope",
            location_anchor_text="parque central",
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="mejor quiero comer algo",
        state=state,
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "restaurant"
    assert intent.reference is None
    assert intent.state_patch.clear_reference is True
    assert intent.state_patch.clear_pending_clarification is True
    assert intent.state_patch.as_dict()["pending_clarification"] is None


def test_common_category_typo_is_normalized_by_the_taxonomy() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="recomiendame una cafetria tranquila",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert "tranquilo" in intent.soft_preferences


def test_pending_scope_can_be_resolved_on_the_following_turn() -> None:
    parser = DeterministicPlaceChatIntentParser()
    first = parser.parse(
        message="cafeterias como la de Hello Kitty cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )
    state = ConversationState(
        target_category=first.target_category,
        soft_preferences=first.soft_preferences,
        exclusions=first.exclusions,
        reference=first.reference,
        pending_clarification=first.state_patch.pending_clarification,
        taxonomy_version="places-taxonomy-v1",
    )

    nearby_results = parser.parse(
        message="la primera opcion",
        state=state,
        has_user_location=True,
    )
    similar_near_user = parser.parse(
        message="la segunda, quiero lugares parecidos",
        state=state,
        has_user_location=True,
    )

    assert nearby_results.action == "recommendations"
    assert nearby_results.location.anchor_text == "parque central"
    assert nearby_results.location.scope == "target_results"
    assert nearby_results.state_patch.clear_pending_clarification is True
    assert similar_near_user.action == "recommendations"
    assert similar_near_user.location.source == "user_current"
    assert similar_near_user.reference is not None
    assert similar_near_user.reference.entity == state.reference.entity
    assert similar_near_user.reference.attributes == state.reference.attributes
    assert similar_near_user.reference.location_hint_text == "parque central"
    assert "hello kitty" in similar_near_user.semantic_query
    assert similar_near_user.state_patch.clear_pending_clarification is True


def test_pending_scope_is_resolved_by_structured_choice_without_repeating() -> None:
    parser = DeterministicPlaceChatIntentParser()
    first = parser.parse(
        message="cafeterias como la de Hello Kitty cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )
    pending = first.state_patch.pending_clarification
    assert pending is not None
    state = ConversationState(
        target_category=first.target_category,
        soft_preferences=first.soft_preferences,
        exclusions=first.exclusions,
        reference=first.reference,
        pending_clarification=pending,
    )

    resolved = parser.parse(
        message="Buscar cerca del Parque Central",
        state=state,
        has_user_location=True,
        clarification_choice=ClarificationChoice(
            clarification_id=pending.clarification_id,
            option_id="target_results",
        ),
    )

    assert resolved.action == "recommendations"
    assert resolved.clarification is None
    assert resolved.state_patch.clear_pending_clarification is True
    assert resolved.location.anchor_text == "parque central"


def test_structured_choice_rejects_a_stale_clarification_id() -> None:
    parser = DeterministicPlaceChatIntentParser()
    first = parser.parse(
        message="quiero un lugar bonito para salir",
        state=ConversationState(),
        has_user_location=True,
    )
    pending = first.state_patch.pending_clarification
    assert pending is not None

    with pytest.raises(ClarificationStateMismatchError):
        parser.parse(
            message="Restaurantes",
            state=ConversationState(pending_clarification=pending),
            has_user_location=True,
            clarification_choice=ClarificationChoice(
                clarification_id="00000000-0000-4000-8000-000000000000",
                option_id="restaurant",
            ),
        )


def test_unrecognized_legacy_reply_keeps_the_same_structured_options() -> None:
    parser = DeterministicPlaceChatIntentParser()
    first = parser.parse(
        message="cafeterias como la de Hello Kitty cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )
    pending = first.state_patch.pending_clarification
    assert pending is not None

    repeated = parser.parse(
        message="no se",
        state=ConversationState(
            target_category=first.target_category,
            reference=first.reference,
            pending_clarification=pending,
        ),
        has_user_location=True,
    )

    assert repeated.action == "clarification"
    assert repeated.clarification is not None
    assert repeated.clarification.clarification_id == pending.clarification_id
    assert len(repeated.clarification.options) == 2


def test_exclusion_is_kept_out_of_positive_preferences() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="una cafeteria tematica pero sin musica",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert "tematica" in intent.soft_preferences
    assert "musica" not in intent.soft_preferences
    assert intent.exclusions == ("musica",)
    assert intent.state_patch.as_dict()["exclusions"] == ["musica"]
