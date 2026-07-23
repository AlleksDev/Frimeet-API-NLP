import pytest

from app.modules.places.domain.chat_intent import (
    ClarificationChoice,
    ConversationState,
    ExplicitTargetLocation,
    PendingClarification,
    PendingClarificationOption,
    PlaceCategoryInference,
    PlaceReference,
)
from app.modules.places.infrastructure.bert_intent_extractor import (
    BertPlaceIntentExtractor,
)
from app.modules.places.domain.errors import ClarificationStateMismatchError
from app.modules.places.infrastructure.deterministic_intent_parser import (
    DeterministicPlaceChatIntentParser,
    PlaceChatTaxonomy,
    PreferenceDefinition,
    load_place_chat_taxonomy,
)


@pytest.mark.parametrize(
    "message",
    ("ola", "hola", "holi", "buenas", "¿qué tal?"),
)
def test_standalone_greetings_return_a_friendly_non_search_response(
    message: str,
) -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "no_match"
    assert intent.unresolved == ("non_search_input",)
    assert intent.response_message is not None
    assert "hola" in intent.response_message.casefold()
    assert "lugar o actividad" in intent.response_message.casefold()
    assert intent.semantic_query == ""
    assert intent.alternatives == ()


def test_greeting_with_a_place_request_remains_actionable() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="hola, busco una cafeteria tranquila",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert "tranquilo" in intent.soft_preferences
    assert "non_search_input" not in intent.unresolved


@pytest.mark.parametrize(
    "message",
    (
        "quiero ir por baguettes",
        "busco un lugar que venda baguete",
        "donde venden baggets",
    ),
)
def test_baguette_variants_resolve_to_bakery_without_losing_product_text(
    message: str,
) -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "bakery"
    assert intent.raw_category_phrase
    assert any(
        term in intent.semantic_query
        for term in ("baguette", "baguete", "baggets")
    )


def test_greeting_repeats_the_same_pending_clarification() -> None:
    pending = PendingClarification(
        clarification_id="pending-category-1",
        kind="intent_category",
        options=(
            PendingClarificationOption(
                option_id="cafe",
                value="cafe",
                label="Cafetería",
            ),
            PendingClarificationOption(
                option_id="restaurant",
                value="restaurant",
                label="Restaurante",
            ),
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="hola",
        state=ConversationState(pending_clarification=pending),
        has_user_location=True,
    )

    assert intent.action == "clarification"
    assert intent.clarification is not None
    assert intent.clarification.clarification_id == pending.clarification_id
    assert [option.option_id for option in intent.clarification.options] == [
        "cafe",
        "restaurant",
    ]
    repeated = intent.state_patch.pending_clarification
    assert repeated is not None
    assert repeated.clarification_id == pending.clarification_id
    assert repeated.options == pending.options
    assert intent.response_message is not None
    assert intent.response_message.startswith("¡Hola!")


def test_transient_place_ids_are_removed_from_filters_and_state_patch() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="una cafeteria",
        state=ConversationState(
            hard_filters={
                "city": "Puebla",
                "place_ids": ("stale-nearby-id",),
            },
        ),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.hard_filters == {"city": "Puebla"}
    assert intent.state_patch.hard_filters == {"city": "Puebla"}
    assert intent.state_patch.as_dict()["hard_filters"] == {"city": "Puebla"}


def test_category_is_not_polluted_by_the_location_anchor() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="Recomiendame alguna cafeteria cerca del Parque Central",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert intent.category_values[:5] == (
        "cafe",
        "café",
        "cafeteria",
        "coffee_shop",
        "coffee shop",
    )
    assert "juice_bar" in intent.category_values
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


def test_new_open_activity_does_not_inherit_previous_category() -> None:
    state = ConversationState(
        target_category="park",
        hard_filters={"price_preference": "lower"},
        soft_preferences=("naturaleza",),
        exclusions=("ruido",),
        reference=PlaceReference(entity="parque central"),
        explicit_target_location=ExplicitTargetLocation(
            anchor_text="parque central"
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero ir a nadar",
        state=state,
        has_user_location=True,
    )

    patch = intent.state_patch.as_dict()
    assert intent.action == "recommendations"
    assert intent.target_category is None
    assert intent.category_source == "unresolved"
    assert "nadar" in intent.semantic_query
    assert "park" not in intent.semantic_query
    assert "parque" not in intent.semantic_query
    assert "naturaleza" not in intent.soft_preferences
    assert "ruido" not in intent.exclusions
    assert intent.reference is None
    assert intent.location.source == "conversation_state"
    assert patch["target_category"] is None
    assert patch["hard_filters"] == {}
    assert patch["soft_preferences"] == list(intent.soft_preferences)
    assert patch["exclusions"] == []
    assert patch["reference"] is None
    assert "explicit_target_location" not in patch


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
    assert "explicit_target_location" not in intent.state_patch.as_dict()


def test_explicit_current_location_replaces_a_stored_anchor() -> None:
    state = ConversationState(
        target_category="cafe",
        explicit_target_location=ExplicitTargetLocation(
            anchor_text="parque central"
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="que sea mas barato cerca de mi",
        state=state,
        has_user_location=True,
    )

    assert intent.target_category == "cafe"
    assert intent.location.source == "user_current"
    assert intent.state_patch.as_dict()["explicit_target_location"] is None


def test_open_activity_then_explicit_category_drops_stale_activity_preference() -> None:
    parser = DeterministicPlaceChatIntentParser()
    swimming = parser.parse(
        message="quiero ir a nadar",
        state=ConversationState(),
        has_user_location=True,
    )
    state = ConversationState(
        target_category=swimming.target_category,
        soft_preferences=swimming.soft_preferences,
    )

    cafe = parser.parse(
        message="ahora quiero una cafeteria tranquila",
        state=state,
        has_user_location=True,
    )

    assert cafe.target_category == "cafe"
    assert "tranquilo" in cafe.soft_preferences
    assert "para_refrescarse" not in cafe.soft_preferences
    assert "para_refrescarse" not in cafe.semantic_query


def test_new_search_for_same_category_clears_old_theme_and_reference() -> None:
    state = ConversationState(
        target_category="cafe",
        soft_preferences=("hello_kitty", "tematica"),
        reference=PlaceReference(entity="hello kitty"),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="ahora quiero una cafeteria tranquila",
        state=state,
        has_user_location=True,
    )

    assert intent.target_category == "cafe"
    assert intent.reference is None
    assert intent.soft_preferences == ("tranquilo",)
    patch = intent.state_patch.as_dict()
    assert patch["reference"] is None
    assert patch["soft_preferences"] == ["tranquilo"]


@pytest.mark.parametrize(
    "message",
    ("dame una opcion mas barata", "recomiendame otra"),
)
def test_common_continuation_commands_keep_the_current_category(
    message: str,
) -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(target_category="park"),
        has_user_location=True,
    )

    assert intent.target_category == "park"
    assert intent.category_source == "conversation_state"


@pytest.mark.parametrize(
    ("message", "preference"),
    (
        ("estacionamiento", "estacionamiento"),
        ("banos", "banos"),
        ("tenga banos", "banos"),
        ("entretenimiento", "entretenimiento"),
        ("pantallas", "pantallas"),
        ("asientos", "asientos"),
        ("comida exterior", "comida_exterior"),
        ("mochila", "mochila"),
    ),
)
def test_bare_attribute_requests_are_preserved_as_preferences(
    message: str,
    preference: str,
) -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert preference in intent.soft_preferences


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


def test_missing_category_preserves_open_concept_for_retrieval() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero un lugar bonito para salir",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category is None
    assert "bonito" in intent.semantic_query
    assert intent.unresolved == ("target_category",)
    assert intent.state_patch.pending_clarification is None


def test_unknown_category_word_is_not_dropped_before_dense_retrieval() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero donas artesanales",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category is None
    assert "donas" in intent.semantic_query
    assert intent.raw_category_phrase is not None


def test_food_activity_defaults_to_restaurant_without_clarification() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero comer algo",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "restaurant"
    assert {
        "restaurant",
        "restaurante",
        "comedor",
        "fast_food_restaurant",
        "taqueria",
        "seafood_restaurant",
    } <= set(intent.category_values)
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
    assert intent.category_values[:5] == (
        "cafe",
        "café",
        "cafeteria",
        "coffee_shop",
        "coffee shop",
    )
    assert "juice_bar" in intent.category_values
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


def test_new_open_activity_cancels_a_stale_pending_clarification() -> None:
    state = ConversationState(
        target_category="park",
        soft_preferences=("naturaleza",),
        pending_clarification=PendingClarification(
            clarification_id="pending-location-scope",
            kind="location_scope",
            location_anchor_text="parque central",
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="quiero ir a nadar",
        state=state,
        has_user_location=True,
    )

    patch = intent.state_patch.as_dict()
    assert intent.action == "recommendations"
    assert intent.clarification is None
    assert intent.target_category is None
    assert "nadar" in intent.semantic_query
    assert "park" not in intent.semantic_query
    assert "parque" not in intent.semantic_query
    assert patch["target_category"] is None
    assert patch["pending_clarification"] is None


@pytest.mark.parametrize("message", ("nadar", "piscina"))
def test_short_open_activity_escapes_a_stale_pending_clarification(
    message: str,
) -> None:
    state = ConversationState(
        target_category="park",
        pending_clarification=PendingClarification(
            clarification_id="pending-category",
            kind="target_category",
            options=(
                PendingClarificationOption(
                    option_id="park",
                    value="park",
                    label="Parques",
                ),
                PendingClarificationOption(
                    option_id="cafe",
                    value="cafe",
                    label="Cafeterías",
                ),
            ),
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=state,
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.clarification is None
    assert intent.target_category is None
    assert intent.state_patch.as_dict()["pending_clarification"] is None


def test_refreshing_preference_is_not_promoted_to_sports_category() -> None:
    base_taxonomy = load_place_chat_taxonomy()
    taxonomy = PlaceChatTaxonomy(
        version=base_taxonomy.version,
        categories=base_taxonomy.categories,
        preferences=(
            *base_taxonomy.preferences,
            PreferenceDefinition(
                canonical="para_refrescarse",
                aliases=("nadar", "natacion", "alberca", "piscina"),
                implies=(),
            ),
        ),
    )

    class SportsClassifier:
        def classify(self, _text: str) -> PlaceCategoryInference | None:
            return PlaceCategoryInference(
                category="sports",
                confidence=0.99,
                source="semantic_activity",
                category_values=("sports",),
            )

        def rank_supported(self, _text: str, limit: int = 3) -> tuple[object, ...]:
            return ()

    intent = DeterministicPlaceChatIntentParser(
        taxonomy=taxonomy,
        activity_classifier=SportsClassifier(),
    ).parse(
        message="quiero ir a nadar",
        state=ConversationState(target_category="park"),
        has_user_location=True,
    )

    assert intent.target_category is None
    assert intent.category_source == "unresolved"
    assert "para_refrescarse" in intent.soft_preferences
    assert "sports" not in intent.semantic_query
    assert intent.state_patch.as_dict()["target_category"] is None


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
        message="cafeterias como la de Hello Kitty cerca del Parque Central",
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


def test_unknown_exclusion_is_preserved_and_does_not_consume_positive_context() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="una cafeteria sin ruido con terraza",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.exclusions == ("ruido",)
    assert "terraza" in intent.semantic_query
    assert "ruido" not in intent.semantic_query


def test_bert_slots_drive_open_category_exclusion_and_location_before_legacy() -> None:
    message = "quiero donas artesanales sin ruido cerca de la plaza"

    def token(value: str, entity: str, score: float) -> dict[str, object]:
        start = message.index(value)
        return {
            "entity": entity,
            "score": score,
            "start": start,
            "end": start + len(value),
        }

    class Classifier:
        def __call__(self, _text: str) -> list[dict[str, object]]:
            return [
                token("donas", "B-CATEGORY", 0.96),
                token("artesanales", "I-CATEGORY", 0.94),
                token("ruido", "B-EXCLUSION", 0.95),
                token("plaza", "B-LOCATION", 0.93),
            ]

    extractor = BertPlaceIntentExtractor(
        "places-intent-test",
        model_version="test-v1",
        classifier=Classifier(),
    )
    intent = DeterministicPlaceChatIntentParser(
        contextual_extractor=extractor,
    ).parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "donas artesanales"
    assert intent.category_values == ("donas artesanales",)
    assert intent.raw_category_phrase == "donas artesanales"
    assert intent.exclusions == ("ruido",)
    assert intent.location.scope == "target_results"
    assert intent.location.anchor_text == "plaza"
    assert "donas artesanales" in intent.semantic_query
    assert "ruido" not in intent.semantic_query
    assert intent.state_patch.target_category == "donas artesanales"
    assert intent.intent_model_version == (
        "bert-token:places-intent-test@test-v1+deterministic-open-v3"
    )


def test_bert_raw_category_is_aligned_semantically_to_dynamic_storage_values() -> None:
    message = "quiero donas artesanales"
    start = message.index("donas")

    class TokenClassifier:
        def __call__(self, _text: str) -> list[dict[str, object]]:
            return [
                {
                    "entity": "B-CATEGORY",
                    "score": 0.96,
                    "start": start,
                    "end": len(message),
                }
            ]

    class Concept:
        id = "donut_shop"
        storage_values = ("bakery", "dessert")

    class ActivityClassifier:
        concepts = (Concept(),)

        def classify(self, text: str) -> PlaceCategoryInference | None:
            assert text == "donas artesanales"
            return PlaceCategoryInference(
                category="donut_shop",
                confidence=0.91,
                source="semantic_activity",
                category_values=("bakery", "dessert"),
                label="Donas",
            )

    intent = DeterministicPlaceChatIntentParser(
        activity_classifier=ActivityClassifier(),
        contextual_extractor=BertPlaceIntentExtractor(
            "places-intent-test",
            classifier=TokenClassifier(),
        ),
    ).parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.target_category == "donut_shop"
    assert intent.category_values == ("bakery", "dessert")
    assert intent.raw_category_phrase == "donas artesanales"


def test_bert_failure_falls_open_and_exposes_fallback_model_version() -> None:
    class BrokenClassifier:
        def __call__(self, _text: str) -> list[dict[str, object]]:
            raise RuntimeError("inference backend unavailable")

    extractor = BertPlaceIntentExtractor(
        "broken-intent-model",
        model_version="broken-v1",
        classifier=BrokenClassifier(),
    )
    intent = DeterministicPlaceChatIntentParser(
        contextual_extractor=extractor,
    ).parse(
        message="recomiendame una cafeteria tranquila",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "cafe"
    assert "tranquilo" in intent.soft_preferences
    assert intent.intent_model_version == (
        "deterministic-open-v3+bert-fallback:broken-v1"
    )


def test_successful_bert_frame_does_not_reapply_manual_category_aliases() -> None:
    extractor = BertPlaceIntentExtractor(
        "places-intent-test",
        classifier=lambda _text: [],
    )

    intent = DeterministicPlaceChatIntentParser(
        contextual_extractor=extractor,
    ).parse(
        message="cafeteria con una variacion no etiquetada",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.target_category is None
    assert intent.category_values == ()
    assert "cafeteria" in intent.semantic_query
    assert intent.category_source == "unresolved"


def test_partial_bert_frame_fuses_missing_context_slots_independently() -> None:
    message = "cafeteria tranquila sin ruido cerca del centro"
    category_start = message.index("cafeteria")
    extractor = BertPlaceIntentExtractor(
        "places-intent-test",
        classifier=lambda _text: [
            {
                "entity": "B-CATEGORY",
                "score": 0.97,
                "start": category_start,
                "end": category_start + len("cafeteria"),
            }
        ],
    )

    intent = DeterministicPlaceChatIntentParser(
        contextual_extractor=extractor,
    ).parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    # The model-provided open category is preserved (no taxonomy remap), while
    # omitted slots use narrow rollout fallbacks instead of disappearing.
    assert intent.target_category == "cafeteria"
    assert intent.location.scope == "target_results"
    assert intent.location.source == "current_message"
    assert intent.location.anchor_text == "centro"
    assert intent.exclusions == ("ruido",)
    assert "tranquilo" in intent.soft_preferences
    assert "centro" not in intent.semantic_query
    assert "ruido" not in intent.semantic_query


def test_bert_preference_reference_and_radius_remain_open_raw_signals() -> None:
    message = "busco salones de te estilo Casa Azul con terraza maximo 2 km"

    def token(value: str, entity: str) -> dict[str, object]:
        start = message.index(value)
        return {
            "entity": entity,
            "score": 0.94,
            "start": start,
            "end": start + len(value),
        }

    class Classifier:
        def __call__(self, _text: str) -> list[dict[str, object]]:
            return [
                token("salones", "B-CATEGORY"),
                token("de", "I-CATEGORY"),
                token("te", "I-CATEGORY"),
                token("Casa", "B-REFERENCE"),
                token("Azul", "I-REFERENCE"),
                token("terraza", "B-PREFERENCE"),
                token("2", "B-RADIUS"),
                token("km", "I-RADIUS"),
            ]

    intent = DeterministicPlaceChatIntentParser(
        contextual_extractor=BertPlaceIntentExtractor(
            "places-intent-test",
            classifier=Classifier(),
        ),
    ).parse(
        message=message,
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.target_category == "salones de te"
    assert intent.soft_preferences == ("terraza",)
    assert intent.reference is not None
    assert intent.reference.entity == "Casa Azul"
    assert intent.location.source == "user_current"
    assert intent.location.radius_meters == 2000
    assert intent.location.strict_radius is True


def test_dynamic_category_clarification_selection_is_not_taxonomy_gated() -> None:
    pending = PendingClarification(
        clarification_id="dynamic-category-1",
        kind="intent_category",
        options=(
            PendingClarificationOption(
                option_id="donuts",
                value="donas artesanales",
                label="Donas artesanales",
            ),
            PendingClarificationOption(
                option_id="desserts",
                value="postres frios",
                label="Postres frios",
            ),
        ),
    )
    state = ConversationState(
        soft_preferences=("tranquilo",),
        exclusions=("ruido",),
        pending_clarification=pending,
        city="Puebla",
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message="Donas artesanales",
        state=state,
        has_user_location=True,
        clarification_choice=ClarificationChoice(
            clarification_id="dynamic-category-1",
            option_id="donuts",
        ),
    )

    assert intent.action == "recommendations"
    assert intent.target_category == "donas artesanales"
    assert intent.category_values == ("donas artesanales",)
    assert intent.raw_category_phrase == "donas artesanales"
    assert intent.soft_preferences == ("tranquilo",)
    assert intent.exclusions == ("ruido",)
    assert intent.hard_filters["city"] == "Puebla"
    assert intent.state_patch.target_category == "donas artesanales"
    assert intent.state_patch.clear_pending_clarification is True
    assert "dynamic-clarification-v1" in intent.intent_model_version


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("la primera opcion", "park"),
        ("la segunda opcion", "cafe"),
        ("no la primera, la segunda", "cafe"),
        ("la primera, no la segunda", "park"),
    ),
)
def test_category_clarification_accepts_textual_ordinal_choice(
    message: str,
    expected: str,
) -> None:
    pending = PendingClarification(
        clarification_id="category-choice-1",
        kind="intent_category",
        options=(
            PendingClarificationOption("park", "park", "Parques"),
            PendingClarificationOption("cafe", "cafe", "Cafeterias"),
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(pending_clarification=pending),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.target_category == expected
    assert intent.state_patch.clear_pending_clarification is True


def test_pending_option_name_inside_a_new_request_is_not_auto_selected() -> None:
    pending = PendingClarification(
        clarification_id="incidental-category-1",
        kind="intent_category",
        options=(
            PendingClarificationOption("park", "park", "Parques"),
            PendingClarificationOption("cafe", "cafe", "Cafeterias"),
        ),
    )
    parser = DeterministicPlaceChatIntentParser()

    restaurant = parser.parse(
        message="quiero un restaurante cerca del cafe central",
        state=ConversationState(pending_clarification=pending),
        has_user_location=True,
    )
    park = parser.parse(
        message="no quiero cafe, quiero un parque",
        state=ConversationState(pending_clarification=pending),
        has_user_location=True,
    )
    cafe_with_pool = parser.parse(
        message="quiero cafe con alberca",
        state=ConversationState(pending_clarification=pending),
        has_user_location=True,
    )

    assert restaurant.target_category == "restaurant"
    assert restaurant.location.anchor_text == "cafe central"
    assert park.target_category == "park"
    assert "cafe" in park.exclusions
    assert cafe_with_pool.target_category == "cafe"
    assert "para_refrescarse" in cafe_with_pool.soft_preferences


@pytest.mark.parametrize("message", ("un zoologico", "rio", "una playa"))
def test_nominal_new_topic_escapes_a_stale_category_clarification(
    message: str,
) -> None:
    pending = PendingClarification(
        clarification_id="stale-category-1",
        kind="intent_category",
        options=(
            PendingClarificationOption("park", "park", "Parques"),
            PendingClarificationOption("cafe", "cafe", "Cafeterias"),
        ),
    )

    intent = DeterministicPlaceChatIntentParser().parse(
        message=message,
        state=ConversationState(
            target_category="park",
            pending_clarification=pending,
        ),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.clarification is None
    assert intent.target_category is None
    assert intent.state_patch.clear_pending_clarification is True
    assert any(term in intent.semantic_query for term in ("zoologico", "rio", "playa"))


def test_open_catalog_category_values_are_rehydrated_on_following_turn() -> None:
    class Concept:
        id = "donut_shop"
        storage_values = ("bakery", "dessert")

    class ActivityClassifier:
        concepts = (Concept(),)

        def classify(self, text: str) -> PlaceCategoryInference | None:
            if "donas" in text:
                return PlaceCategoryInference(
                    category="donut_shop",
                    confidence=0.91,
                    source="semantic_activity",
                    category_values=("bakery", "dessert"),
                    label="Donas",
                )
            return None

    parser = DeterministicPlaceChatIntentParser(
        activity_classifier=ActivityClassifier(),
    )
    first = parser.parse(
        message="quiero donas glaseadas",
        state=ConversationState(),
        has_user_location=True,
    )
    continued = parser.parse(
        message="mas barato",
        state=ConversationState(target_category=first.target_category),
        has_user_location=True,
    )

    assert first.target_category == "donut_shop"
    assert first.category_values == ("bakery", "dessert")
    assert continued.target_category == "donut_shop"
    assert continued.category_values == ("bakery", "dessert")


def test_generic_short_request_never_produces_an_empty_embedding_query() -> None:
    intent = DeterministicPlaceChatIntentParser().parse(
        message="dame opciones",
        state=ConversationState(),
        has_user_location=True,
    )

    assert intent.action == "recommendations"
    assert intent.semantic_query == "dame opciones"
