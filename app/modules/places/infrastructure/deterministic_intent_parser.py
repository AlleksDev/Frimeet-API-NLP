from dataclasses import dataclass, replace
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable

from app.modules.places.application.ports.activity_classifier import (
    PlaceActivityClassifier,
)
from app.modules.places.domain.clarifications import (
    ensure_legacy_pending_options,
    new_category_clarification,
    new_location_scope_clarification,
    to_public_clarification,
)
from app.modules.places.domain.chat_intent import (
    CategoryInferenceSource,
    ClarificationChoice,
    ConversationState,
    ConversationStatePatch,
    ExplicitTargetLocation,
    IntentAlternative,
    LocationIntent,
    PendingClarification,
    PendingClarificationOption,
    PlaceCategoryInference,
    ParsedPlaceChatIntent,
    PlaceReference,
)
from app.modules.places.domain.errors import ClarificationStateMismatchError
from app.modules.places.infrastructure.bert_intent_extractor import (
    BertIntentExtractionError,
    BertPlaceIntentExtractor,
    IntentFrame,
    IntentSpan,
)
from app.shared.nlp.preprocessing.text import (
    prepare_for_embedding,
    tokenize_for_embeddings,
)


_REFERENCE_PATTERN = re.compile(
    r"\b(?:como\s+(?:la|el)\s+de|parecid[oa]s?\s+a|similar(?:es)?\s+a)\s+"
    r"(?P<value>.+?)(?=\s+cerca\s+(?:de\s+la|del|de)\b|[,;]|$)"
)
_LOCATION_PATTERN = re.compile(
    r"\bcerca\s+(?:de\s+los|de\s+las|de\s+la|del|de)\s+"
    r"(?P<value>.+?)(?=[,;]|\s+(?:pero|aunque)\b|$)"
)
_RADIUS_PATTERN = re.compile(
    r"\b(?:a|en\s+un\s+radio\s+de|radio\s+de)\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>km|kilometros?|m|metros?)\b"
)
_EXCLUSION_PATTERN = re.compile(
    r"\b(?:sin|excepto|evita(?:r)?|no\s+quiero|que\s+no\s+(?:sea|tenga))\s+"
    r"(?P<value>.+?)(?="
    r"\s+cerca\s+(?:de\s+la|del|de)\b|[,;]|"
    r"\s+(?:pero|aunque|con)\b|\s+y\s+(?:con|que|quiero|busco)\b|$)"
)
_CURRENT_LOCATION_VALUES = {"mi", "aqui", "donde estoy", "mi ubicacion"}
_DETERMINISTIC_INTENT_VERSION = "deterministic-open-v3"
_RADIUS_VALUE_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>km|kilometros?|m|metros?)\b"
)
_LOGGER = logging.getLogger(__name__)
_TRANSIENT_HARD_FILTER_KEYS = {"place_ids"}
_SOCIAL_GREETING_VALUES = {
    "buenas",
    "buenas noches",
    "buenas tardes",
    "buenos dias",
    "hello",
    "hey",
    "hola",
    "hola como estas",
    "hola que tal",
    "hola que tal como estas",
    "holi",
    "holis",
    "ola",
    "ola que tal",
    "que tal",
}
_SOCIAL_THANKS_VALUES = {
    "gracias",
    "mil gracias",
    "muchas gracias",
    "te lo agradezco",
}
_SOCIAL_FAREWELL_VALUES = {
    "adios",
    "bye",
    "hasta luego",
    "hasta pronto",
    "nos vemos",
}
_SOCIAL_ACKNOWLEDGEMENT_VALUES = {
    "entendido",
    "listo",
    "ok",
    "okay",
    "perfecto",
    "vale",
}
_GENERIC_REQUEST_TOKENS = {
    "dame",
    "favor",
    "opcion",
    "opciones",
    "podrias",
    "quisiera",
    "parecida",
    "parecidas",
    "parecido",
    "parecidos",
    "recomienda",
    "recomendacion",
    "recomendaciones",
    "recomiendame",
    "similar",
    "similares",
}
_OPEN_ACTIVITY_PREFERENCES = frozenset({"para_refrescarse"})
_CONTINUATION_PREFIX_PATTERN = re.compile(
    r"^(?:(?:y|pero)\s+)?"
    r"(?:(?:quiero|dame|recomiendame|recomienda|muestrame)\s+)?"
    r"(?:(?:un|una)\s+)?(?:"
    r"otras?|otros?|mas|menos|tambien|"
    r"opcion(?:es)?(?:\s+(?:mas|barata|barato|cercana|cercano))?|"
    r"que\s+(?:sea|tenga|este|quede|cueste)|"
    r"con|sin|cerca|lejos|aqui|ahi|alrededor|"
    r"la\s+primera|la\s+segunda|opcion\s+(?:uno|dos)"
    r")\b"
)
_NEW_SEARCH_LEAD_PATTERN = re.compile(
    r"^(?:(?:ahora|mejor|cambiando\s+de\s+tema|en\s+lugar\s+de\s+eso)\s+)*"
    r"(?:quiero|quisiera|busco|necesito|prefiero|"
    r"me\s+gustaria|tengo\s+ganas\s+de|se\s+me\s+antoja|"
    r"recomiendame|recomienda|dame|"
    r"donde\s+(?:puedo|hay|encuentro|consigo|ir))\b"
)
_CONTINUATION_REQUEST_PHRASES = (
    "otra opcion",
    "otras opciones",
    "dame mas",
    "muestrame mas",
    "algo parecido",
    "algo similar",
    "parecida",
    "parecidas",
    "parecido",
    "parecidos",
    "similar",
    "similares",
    "lugares parecidos",
    "lugares similares",
    "quiero que sea",
    "quiero que tenga",
)
_GENERIC_CONTINUATION_TOKENS = _GENERIC_REQUEST_TOKENS | {
    "algo",
    "dame",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "me",
    "muestra",
    "muestrame",
    "otra",
    "otras",
    "otro",
    "otros",
    "una",
    "unas",
    "uno",
    "unos",
}

# High-precision activity intents that imply a useful place category even when
# the user does not name it. These defaults keep the chat action-oriented while
# remaining deterministic and auditable; explicit category aliases always win.
_ACTIVITY_CATEGORY_DEFAULTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "restaurant",
        (
            "quiero comer",
            "quisiera comer",
            "me gustaria comer",
            "necesito comer",
            "tengo hambre",
            "algo para comer",
            "algo de comer",
            "comer algo",
            "donde comer",
            "ir a comer",
            "salir a comer",
            "quiero desayunar",
            "quisiera desayunar",
            "quiero almorzar",
            "quisiera almorzar",
            "quiero cenar",
            "quisiera cenar",
            "comida vegana",
            "comida vegetariana",
            "algo vegano",
            "algo vegetariano",
            "algo sin gluten",
        ),
    ),
    (
        "sports",
        (
            "hacer ejercicio",
            "quiero entrenar",
            "quisiera entrenar",
            "donde entrenar",
        ),
    ),
    (
        "cinema",
        (
            "ver una pelicula",
            "ver peliculas",
        ),
    ),
    (
        "shopping",
        (
            "quiero comprar algo",
            "ir de compras",
            "salir de compras",
        ),
    ),
    (
        "lodging",
        (
            "donde dormir",
            "donde hospedarme",
            "quiero hospedarme",
            "necesito alojamiento",
            "pasar la noche",
        ),
    ),
)


@dataclass(frozen=True)
class CategoryDefinition:
    canonical: str
    aliases: tuple[str, ...]
    storage_values: tuple[str, ...]
    compatible_storage_values: tuple[str, ...]
    evidence_terms: tuple[str, ...]


@dataclass(frozen=True)
class PreferenceDefinition:
    canonical: str
    aliases: tuple[str, ...]
    implies: tuple[str, ...]


@dataclass(frozen=True)
class PlaceChatTaxonomy:
    version: str
    categories: tuple[CategoryDefinition, ...]
    preferences: tuple[PreferenceDefinition, ...]

    def category(self, canonical: str | None) -> CategoryDefinition | None:
        if canonical is None:
            return None
        return next(
            (item for item in self.categories if item.canonical == canonical),
            None,
        )


@dataclass(frozen=True)
class _ContextualSignals:
    frame: IntentFrame | None
    category_phrases: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    location: str | None = None
    reference: str | None = None
    radius_meters: int | None = None
    category_confidence: float | None = None
    intent_model_version: str = _DETERMINISTIC_INTENT_VERSION


class DeterministicPlaceChatIntentParser:
    def __init__(
        self,
        taxonomy: PlaceChatTaxonomy | None = None,
        activity_classifier: PlaceActivityClassifier | None = None,
        contextual_extractor: BertPlaceIntentExtractor | None = None,
    ) -> None:
        self._taxonomy = taxonomy or load_place_chat_taxonomy()
        self._activity_classifier = activity_classifier
        self._contextual_extractor = contextual_extractor

    def parse(
        self,
        message: str,
        state: ConversationState,
        has_user_location: bool,
        clarification_choice: ClarificationChoice | None = None,
    ) -> ParsedPlaceChatIntent:
        normalized = prepare_for_embedding(message)
        if clarification_choice is not None:
            return self._resolve_structured_clarification(
                choice=clarification_choice,
                state=state,
                has_user_location=has_user_location,
            )
        social_turn = _social_turn(normalized)
        if social_turn is not None:
            social_kind, social_message = social_turn
            if social_kind == "greeting" and state.pending_clarification is not None:
                repeated = self._clarification(
                    pending=state.pending_clarification,
                    unresolved=(state.pending_clarification.kind,),
                    state=state,
                    has_user_location=has_user_location,
                )
                if repeated.clarification is None:
                    raise RuntimeError("pending clarification requires public options")
                prompt = f"¡Hola! {repeated.clarification.prompt}"
                return replace(
                    repeated,
                    clarification=replace(repeated.clarification, prompt=prompt),
                    clarification_message=prompt,
                    response_message=prompt,
                )
            return self._direct_response(
                message=social_message,
                state=state,
            )
        contextual = self._contextual_signals(message)
        # Token classification is multi-label and a valid frame can still be
        # incomplete.  Strip both model-detected context spans and any remaining
        # high-precision legacy context before building the retrieval query.  The
        # category itself stays model/open-vocabulary first; taxonomy aliases are
        # deliberately not restored for a successful BERT frame.
        target_clause = self._target_clause(
            _remove_contextual_spans(normalized, contextual.frame)
        )
        if state.pending_clarification:
            pending = ensure_legacy_pending_options(state.pending_clarification)
            state = replace(state, pending_clarification=pending)
            selected_pending_option = self._pending_choice_option(
                normalized,
                pending,
            )
            if selected_pending_option is not None:
                resolved = self._resolve_structured_clarification(
                    choice=ClarificationChoice(
                        clarification_id=pending.clarification_id,
                        option_id=selected_pending_option.option_id,
                    ),
                    state=state,
                    has_user_location=has_user_location,
                )
                return replace(
                    resolved,
                    intent_model_version=contextual.intent_model_version,
                )
            categories = self._contextual_categories(contextual, target_clause)
            inferred = (
                self._inferred_activity_category(
                    target_clause,
                    allow_lexical_defaults=contextual.frame is None,
                )
                if not categories
                else None
            )
            if (
                (
                    categories
                    or inferred is not None
                    or self._has_open_activity_preference(normalized)
                    or self._starts_new_search_turn(normalized)
                    or self._has_new_nominal_topic(target_clause)
                )
                and not self._has_explicit_pending_choice(normalized)
            ):
                previous_state = state
                fresh_intent = self.parse(
                    message=message,
                    state=self._without_topic_context(state),
                    has_user_location=has_user_location,
                )
                return replace(
                    self._with_topic_reset(
                        fresh_intent,
                        previous_state=previous_state,
                    ),
                    intent_model_version=contextual.intent_model_version,
                )
            return replace(
                self._resolve_pending_clarification(
                    normalized=normalized,
                    state=state,
                    has_user_location=has_user_location,
                ),
                intent_model_version=contextual.intent_model_version,
            )

        # Fuse slots independently.  A CATEGORY-only frame must not erase a
        # LOCATION, RADIUS, REFERENCE, preference, or exclusion that the model
        # omitted.  These narrow extractors are rollout fallbacks, not gates.
        reference_text = contextual.reference or self._extract_reference(normalized)
        location_text = contextual.location or self._extract_location(normalized)
        radius_meters = (
            contextual.radius_meters
            if contextual.radius_meters is not None
            else self._extract_radius_meters(normalized)
        )
        excluded_texts = tuple(
            match.group("value").strip(" ,.;")
            for match in _EXCLUSION_PATTERN.finditer(normalized)
        )

        # Category words inside references, exclusions, or geographic anchors
        # do not describe the requested result type. In "cafeteria cerca del
        # parque", only cafe is a hard category.
        categories = self._contextual_categories(contextual, target_clause)
        if len(categories) > 1:
            pending = new_category_clarification(categories)
            return replace(
                self._clarification(
                    pending=pending,
                    unresolved=("target_category",),
                    state=state,
                    has_user_location=has_user_location,
                ),
                intent_model_version=contextual.intent_model_version,
            )

        explicit_category = categories[0] if categories else None
        inferred = (
            self._inferred_activity_category(
                target_clause,
                allow_lexical_defaults=contextual.frame is None,
            )
            if explicit_category is None
            else None
        )
        category_alternatives = (
            self._ranked_activity_alternatives(target_clause)
            if explicit_category is None
            else ()
        )
        inferred_category = inferred.category if inferred else None
        requested_category = explicit_category or inferred_category
        previous_state = state
        starts_new_search = self._starts_new_search_turn(normalized)
        continuation_turn = self._is_continuation_turn(
            normalized=normalized,
            target_clause=target_clause,
            contextual=contextual,
        )
        category_changed = bool(
            requested_category
            and state.target_category
            and requested_category != state.target_category
        )
        topic_reset = bool(
            self._has_topic_context(state)
            and (
                category_changed
                or starts_new_search
                or (
                    requested_category is not None
                    and state.target_category is None
                    and not continuation_turn
                )
                or (
                    requested_category is None
                    and not continuation_turn
                )
            )
        )
        if topic_reset:
            state = self._without_topic_context(state)
        target_category = requested_category or state.target_category
        if explicit_category:
            category_source: CategoryInferenceSource = "explicit"
        elif inferred:
            category_source = inferred.source
        elif state.target_category:
            category_source = "conversation_state"
        else:
            category_source = "unresolved"

        if location_text:
            location_text = _RADIUS_PATTERN.sub("", location_text).strip(" ,.;")

        legacy_preference_text = normalized if not contextual.preferences else ""
        detected_preferences = _ordered_unique(
            (
                *contextual.preferences,
                *self._matched_preferences(legacy_preference_text),
            )
        )
        legacy_excluded_texts = tuple(
            excluded_text
            for excluded_text in excluded_texts
            if not any(
                _contains_phrase(
                    prepare_for_embedding(excluded_text),
                    prepare_for_embedding(contextual_exclusion),
                )
                for contextual_exclusion in contextual.exclusions
            )
        )
        explicit_exclusions = _ordered_unique(
            (
                *contextual.exclusions,
                *(
                    exclusion
                    for excluded_text in legacy_excluded_texts
                    for exclusion in (
                        self._matched_preferences(excluded_text)
                        or (prepare_for_embedding(excluded_text),)
                    )
                ),
            )
        )
        positive_preferences = tuple(
            preference
            for preference in detected_preferences
            if preference not in explicit_exclusions
        )

        if category_changed:
            inherited_preferences: tuple[str, ...] = ()
            inherited_exclusions: tuple[str, ...] = ()
            inherited_reference = None
        else:
            inherited_preferences = state.soft_preferences
            inherited_exclusions = state.exclusions
            inherited_reference = state.reference

        merged_preferences = tuple(
            preference
            for preference in _ordered_unique(
                (*inherited_preferences, *positive_preferences)
            )
            if preference not in explicit_exclusions
        )
        merged_exclusions = tuple(
            exclusion
            for exclusion in _ordered_unique(
                (*inherited_exclusions, *explicit_exclusions)
            )
            if exclusion not in positive_preferences
        )

        reference = (
            PlaceReference(
                entity=reference_text,
                attributes=positive_preferences,
            )
            if reference_text
            else inherited_reference
        )
        hard_filters = _persistent_hard_filters(state.hard_filters)
        if state.city and "city" not in hard_filters:
            hard_filters["city"] = state.city
        if state.state and "state" not in hard_filters:
            hard_filters["state"] = state.state
        if "economico" in merged_preferences or _contains_any(
            target_clause,
            ("mas barato", "economico", "barato"),
        ):
            hard_filters["price_preference"] = "lower"

        # An unresolved category is not a reason to stop retrieval.  Preserve the
        # user's open-ended concept in ``semantic_query`` and let dense/lexical
        # retrieval provide evidence.  A downstream decision policy may still ask
        # a clarification, but its options must come from model hypotheses or real
        # candidates rather than a fixed menu.
        category = (
            self._taxonomy.category(target_category)
            if target_category is not None
            else None
        )
        inferred_category_values = (
            tuple(inferred.category_values)
            if inferred is not None
            and inferred.category == target_category
            and inferred.category_values
            else ()
        )
        rehydrated_category_values = self._activity_category_values(
            target_category
        )
        category_values = (
            category.storage_values
            if category is not None
            else (
                inferred_category_values
                or rehydrated_category_values
                or ((target_category,) if target_category else ())
            )
        )
        compatible_category_values = (
            category.compatible_storage_values if category else ()
        )
        category_evidence_terms = category.evidence_terms if category else ()

        if (
            reference_text
            and location_text
            and prepare_for_embedding(location_text) not in _CURRENT_LOCATION_VALUES
        ):
            clarified = replace(
                self._location_scope_clarification(
                    target_category=target_category,
                    category_values=category_values,
                    compatible_category_values=compatible_category_values,
                    category_evidence_terms=category_evidence_terms,
                    explicit_category=requested_category,
                    category_source=category_source,
                    hard_filters=hard_filters,
                    preferences=merged_preferences,
                    exclusions=merged_exclusions,
                    reference=reference,
                    reference_text=reference_text,
                    location_text=location_text,
                    radius_meters=radius_meters,
                    state=state,
                    has_user_location=has_user_location,
                ),
                intent_model_version=contextual.intent_model_version,
            )
            return (
                self._with_topic_reset(
                    clarified,
                    previous_state=previous_state,
                )
                if topic_reset
                else clarified
            )

        location, explicit_location = self._location_intent(
            location_text=location_text,
            radius_meters=radius_meters,
            state=state,
            has_user_location=has_user_location,
        )
        category_alias_tokens: set[str] = set()
        if explicit_category and category:
            for alias in (category.canonical, *category.aliases):
                category_alias_tokens.update(tokenize_for_embeddings(alias))
        semantic_target = " ".join(
            token
            for token in tokenize_for_embeddings(target_clause)
            if token not in _GENERIC_REQUEST_TOKENS
            and token not in category_alias_tokens
        )
        category_query_term = self._category_query_term(target_clause, category)
        semantic_parts = [
            target_category or "",
            category_query_term,
            semantic_target,
            *merged_preferences,
        ]
        if reference and reference.entity:
            semantic_parts.append(reference.entity)
        semantic_query = (
            " ".join(_ordered_unique(semantic_parts)).strip()
            or target_clause
            or normalized
        )

        patch = ConversationStatePatch(
            target_category=requested_category,
            hard_filters=hard_filters if hard_filters != state.hard_filters else None,
            soft_preferences=(
                merged_preferences
                if merged_preferences != state.soft_preferences
                else None
            ),
            exclusions=(
                merged_exclusions if merged_exclusions != state.exclusions else None
            ),
            reference=reference if reference_text else None,
            explicit_target_location=explicit_location,
            clear_reference=category_changed and reference_text is None,
            clear_explicit_target_location=(
                previous_state.explicit_target_location is not None
                and location_text is not None
                and location.source == "user_current"
            ),
            taxonomy_version=self._taxonomy.version,
        )
        if explicit_category and contextual.category_phrases:
            confidence = contextual.category_confidence or contextual.frame.confidence
        elif explicit_category:
            confidence = 0.96
        elif inferred_category:
            confidence = inferred.confidence if inferred else 0.88
        elif target_category is None:
            # This confidence describes category resolution, not whether the
            # query is searchable. Keeping it below the automatic-decision band
            # exposes uncertainty while retrieval can still return evidence.
            confidence = 0.55
        else:
            confidence = 0.84
        if reference_text:
            confidence -= 0.05
        parsed = ParsedPlaceChatIntent(
            action="recommendations",
            target_category=target_category,
            category_values=category_values,
            hard_filters=hard_filters,
            soft_preferences=merged_preferences,
            exclusions=merged_exclusions,
            reference=reference,
            location=location,
            semantic_query=semantic_query,
            confidence=max(0.0, min(1.0, confidence)),
            state_patch=patch,
            compatible_category_values=compatible_category_values,
            category_evidence_terms=category_evidence_terms,
            category_source=category_source,
            alternatives=category_alternatives,
            unresolved=("target_category",) if target_category is None else (),
            raw_category_phrase=(
                contextual.category_phrases[0]
                if contextual.category_phrases
                else (
                    semantic_target or target_clause
                    if explicit_category is None
                    else explicit_category
                )
            ),
            intent_model_version=contextual.intent_model_version,
        )
        return (
            self._with_topic_reset(parsed, previous_state=previous_state)
            if topic_reset
            else parsed
        )

    @staticmethod
    def _without_topic_context(state: ConversationState) -> ConversationState:
        return replace(
            state,
            target_category=None,
            hard_filters={
                key: value
                for key, value in state.hard_filters.items()
                if key in {"city", "state"}
            },
            soft_preferences=(),
            exclusions=(),
            reference=None,
            pending_clarification=None,
        )

    @staticmethod
    def _has_topic_context(state: ConversationState) -> bool:
        topic_filter_keys = set(state.hard_filters) - {"city", "state", "place_ids"}
        return bool(
            state.target_category
            or topic_filter_keys
            or state.soft_preferences
            or state.exclusions
            or state.reference
            or state.explicit_target_location
            or state.pending_clarification
        )

    @staticmethod
    def _with_topic_reset(
        intent: ParsedPlaceChatIntent,
        *,
        previous_state: ConversationState,
    ) -> ParsedPlaceChatIntent:
        patch = intent.state_patch
        return replace(
            intent,
            state_patch=replace(
                patch,
                clear_target_category=(
                    patch.clear_target_category
                    or (
                        previous_state.target_category is not None
                        and intent.target_category is None
                        and patch.target_category is None
                    )
                ),
                hard_filters=(
                    dict(intent.hard_filters)
                    if intent.hard_filters != previous_state.hard_filters
                    else patch.hard_filters
                ),
                soft_preferences=(
                    intent.soft_preferences
                    if intent.soft_preferences != previous_state.soft_preferences
                    else patch.soft_preferences
                ),
                exclusions=(
                    intent.exclusions
                    if intent.exclusions != previous_state.exclusions
                    else patch.exclusions
                ),
                clear_reference=(
                    patch.clear_reference
                    or (
                        previous_state.reference is not None
                        and intent.reference is None
                        and patch.reference is None
                    )
                ),
                clear_explicit_target_location=(
                    patch.clear_explicit_target_location
                    or (
                        previous_state.explicit_target_location is not None
                        and patch.explicit_target_location is None
                        and intent.location.source != "conversation_state"
                    )
                ),
                clear_pending_clarification=(
                    patch.clear_pending_clarification
                    or (
                        previous_state.pending_clarification is not None
                        and patch.pending_clarification is None
                    )
                ),
            ),
        )

    def _has_open_activity_preference(self, normalized: str) -> bool:
        return any(
            preference.canonical in _OPEN_ACTIVITY_PREFERENCES
            and any(
                _contains_phrase(normalized, alias)
                for alias in preference.aliases
            )
            for preference in self._taxonomy.preferences
        )

    @staticmethod
    def _starts_new_search_turn(normalized: str) -> bool:
        standalone = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        if not standalone:
            return False
        if _contains_any(standalone, _CONTINUATION_REQUEST_PHRASES):
            return False
        if _CONTINUATION_PREFIX_PATTERN.search(standalone):
            return False
        return bool(_NEW_SEARCH_LEAD_PATTERN.search(standalone))

    @staticmethod
    def _has_new_nominal_topic(target_clause: str) -> bool:
        tokens = set(tokenize_for_embeddings(target_clause))
        ignored = _GENERIC_CONTINUATION_TOKENS | {
            "cerca",
            "esa",
            "ese",
            "identificar",
            "no",
            "opcion",
            "primera",
            "referencia",
            "se",
            "segunda",
            "solo",
            "zona",
        }
        return bool(tokens - ignored)

    def _is_continuation_turn(
        self,
        *,
        normalized: str,
        target_clause: str,
        contextual: _ContextualSignals,
    ) -> bool:
        if self._has_open_activity_preference(normalized):
            return False
        if self._starts_new_search_turn(normalized):
            return False

        standalone = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        if _contains_any(standalone, _CONTINUATION_REQUEST_PHRASES):
            return True
        if _CONTINUATION_PREFIX_PATTERN.search(standalone):
            return True
        if (
            contextual.reference
            or contextual.location
            or contextual.radius_meters is not None
            or contextual.exclusions
            or self._extract_reference(normalized)
            or self._extract_location(normalized)
            or self._extract_radius_meters(normalized) is not None
            or _EXCLUSION_PATTERN.search(normalized)
        ):
            return True
        if self._matched_preferences(normalized):
            return True

        tokens = set(tokenize_for_embeddings(target_clause))
        return bool(tokens) and tokens <= _GENERIC_CONTINUATION_TOKENS

    def _direct_response(
        self,
        message: str,
        state: ConversationState,
    ) -> ParsedPlaceChatIntent:
        hard_filters = _persistent_hard_filters(state.hard_filters)
        return ParsedPlaceChatIntent(
            action="no_match",
            target_category=None,
            category_values=(),
            hard_filters=hard_filters,
            soft_preferences=(),
            exclusions=(),
            reference=None,
            location=LocationIntent(scope="unresolved", source="none"),
            semantic_query="",
            confidence=1.0,
            state_patch=ConversationStatePatch(
                hard_filters=(
                    hard_filters if hard_filters != state.hard_filters else None
                ),
                taxonomy_version=self._taxonomy.version,
            ),
            unresolved=("non_search_input",),
            response_message=message,
            intent_model_version=_DETERMINISTIC_INTENT_VERSION,
        )

    def _location_scope_clarification(
        self,
        target_category: str,
        category_values: tuple[str, ...],
        compatible_category_values: tuple[str, ...],
        category_evidence_terms: tuple[str, ...],
        explicit_category: str | None,
        category_source: CategoryInferenceSource,
        hard_filters: dict[str, Any],
        preferences: tuple[str, ...],
        exclusions: tuple[str, ...],
        reference: PlaceReference | None,
        reference_text: str,
        location_text: str,
        radius_meters: int | None,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        pending = new_location_scope_clarification(
            anchor_text=location_text,
            radius_meters=radius_meters,
        )
        clarification = to_public_clarification(pending)
        return ParsedPlaceChatIntent(
            action="clarification",
            target_category=target_category,
            category_values=category_values,
            hard_filters=hard_filters,
            soft_preferences=preferences,
            exclusions=exclusions,
            reference=reference,
            location=(
                LocationIntent(
                    scope="user_current_location",
                    source="user_current",
                )
                if has_user_location
                else LocationIntent(scope="unresolved", source="none")
            ),
            semantic_query="",
            confidence=0.75,
            state_patch=ConversationStatePatch(
                target_category=explicit_category,
                hard_filters=(
                    hard_filters if hard_filters != state.hard_filters else None
                ),
                soft_preferences=(
                    preferences if preferences != state.soft_preferences else None
                ),
                exclusions=(
                    exclusions if exclusions != state.exclusions else None
                ),
                reference=reference,
                pending_clarification=pending,
                taxonomy_version=self._taxonomy.version,
            ),
            compatible_category_values=compatible_category_values,
            category_evidence_terms=category_evidence_terms,
            category_source=category_source,
            clarification=clarification,
            alternatives=(
                IntentAlternative(
                    key="target_results",
                    description=f"Usar {location_text} para ubicar los resultados",
                    confidence=0.5,
                ),
                IntentAlternative(
                    key="reference_entity",
                    description=(
                        f"Usar {location_text} solo para identificar la referencia"
                    ),
                    confidence=0.5,
                ),
            ),
            unresolved=("location_scope",),
            clarification_message=clarification.prompt,
        )

    def _resolve_structured_clarification(
        self,
        choice: ClarificationChoice,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        pending = state.pending_clarification
        if pending is None:
            raise ClarificationStateMismatchError(
                "there is no pending clarification"
            )
        pending = ensure_legacy_pending_options(pending)
        if choice.clarification_id != pending.clarification_id:
            raise ClarificationStateMismatchError(
                "clarification id does not match the pending state"
            )
        selected = next(
            (
                option
                for option in pending.options
                if option.option_id == choice.option_id
            ),
            None,
        )
        if selected is None:
            raise ClarificationStateMismatchError(
                "clarification option is not allowed"
            )

        base_state = replace(state, pending_clarification=None)
        if pending.kind in ("target_category", "intent_category"):
            if self._taxonomy.category(selected.value) is None:
                return self._resolve_open_category_choice(
                    value=selected.value,
                    state=base_state,
                    has_user_location=has_user_location,
                )
            resolved = self.parse(
                message=self._category_message(selected.value),
                state=base_state,
                has_user_location=has_user_location,
            )
            return self._with_cleared_pending(resolved)

        if pending.kind == "location_scope":
            return self._resolve_location_scope_value(
                value=selected.value,
                pending=pending,
                state=state,
                has_user_location=has_user_location,
            )

        if pending.kind == "location_anchor":
            if not selected.place_id:
                raise ClarificationStateMismatchError(
                    "selected location anchor has no place id"
                )
            explicit = ExplicitTargetLocation(
                anchor_text=pending.location_anchor_text or selected.label,
                place_id=selected.place_id,
                label=selected.label,
                radius_meters=pending.radius_meters,
                strict_radius=pending.strict_radius,
            )
            resolved = self.parse(
                message=self._category_message(state.target_category),
                state=replace(base_state, explicit_target_location=explicit),
                has_user_location=has_user_location,
            )
            return replace(
                resolved,
                location=LocationIntent(
                    scope="target_results",
                    source="current_message",
                    anchor_text=explicit.anchor_text,
                    resolved_place_id=explicit.place_id,
                    radius_meters=explicit.radius_meters,
                    strict_radius=explicit.strict_radius,
                ),
                state_patch=replace(
                    resolved.state_patch,
                    explicit_target_location=explicit,
                    clear_pending_clarification=True,
                ),
            )

        if pending.kind == "reference_entity":
            if not selected.place_id or state.reference is None:
                raise ClarificationStateMismatchError(
                    "selected reference cannot be applied"
                )
            reference = replace(
                state.reference,
                place_id=selected.place_id,
                attributes=tuple(
                    dict.fromkeys((*state.reference.attributes, *selected.attributes))
                ),
            )
            resolved = self.parse(
                message=self._category_message(state.target_category),
                state=replace(base_state, reference=reference),
                has_user_location=has_user_location,
            )
            return replace(
                resolved,
                state_patch=replace(
                    resolved.state_patch,
                    reference=reference,
                    clear_pending_clarification=True,
                ),
            )

        if pending.kind == "reference_location_anchor":
            if not selected.place_id or state.reference is None:
                raise ClarificationStateMismatchError(
                    "selected reference location cannot be applied"
                )
            reference = replace(
                state.reference,
                location_hint_text=pending.location_anchor_text or selected.label,
                location_hint_place_id=selected.place_id,
            )
            resolved = self.parse(
                message=self._category_message(state.target_category),
                state=replace(base_state, reference=reference),
                has_user_location=has_user_location,
            )
            return replace(
                resolved,
                state_patch=replace(
                    resolved.state_patch,
                    reference=reference,
                    clear_pending_clarification=True,
                ),
            )

        raise ClarificationStateMismatchError(
            "unsupported pending clarification kind"
        )

    def _resolve_pending_clarification(
        self,
        normalized: str,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        pending = state.pending_clarification
        if pending is None:
            raise RuntimeError("pending clarification is required")

        pending = ensure_legacy_pending_options(pending)
        selected = self._pending_choice_option(normalized, pending)
        if selected is not None:
            return self._resolve_structured_clarification(
                choice=ClarificationChoice(
                    clarification_id=pending.clarification_id,
                    option_id=selected.option_id,
                ),
                state=replace(state, pending_clarification=pending),
                has_user_location=has_user_location,
            )

        reference_choice = _contains_any(
            normalized,
            (
                "segunda",
                "opcion dos",
                "otra zona",
                "aunque esten en otra zona",
                "parecidos",
                "solo para identificar",
                "es la referencia",
                "cerca de mi",
                "mi ubicacion",
                "donde estoy",
            ),
        )
        target_choice = _contains_any(
            normalized,
            (
                "primera",
                "opcion uno",
                "cerca",
                "esa zona",
                "ese lugar",
                "alrededor",
                "ahi",
            ),
        )
        if not reference_choice and not target_choice:
            return self._clarification(
                pending=pending,
                unresolved=(pending.kind,),
                state=state,
                has_user_location=has_user_location,
            )
        return self._resolve_location_scope_value(
            value="reference_entity" if reference_choice else "target_results",
            pending=pending,
            state=state,
            has_user_location=has_user_location,
        )

    @staticmethod
    def _pending_choice_option(
        normalized: str,
        pending: PendingClarification,
    ) -> PendingClarificationOption | None:
        if not pending.options:
            return None

        first_position = _last_affirmed_phrase_position(
            normalized,
            ("primera", "opcion uno"),
        )
        second_position = _last_affirmed_phrase_position(
            normalized,
            ("segunda", "opcion dos"),
        )
        if first_position >= 0 or second_position >= 0:
            if second_position > first_position:
                return pending.options[1] if len(pending.options) > 1 else None
            return pending.options[0]

        choice_text = re.sub(
            r"^(?:(?:elijo|escojo|selecciono|prefiero|quiero)\s+)?"
            r"(?:(?:la|el|un|una)\s+)?(?:opcion\s+)?(?:de\s+)?",
            "",
            normalized,
        ).strip()
        for option in pending.options:
            for raw_value in (option.label, option.value, option.option_id):
                value = prepare_for_embedding(raw_value)
                if value and choice_text == value:
                    return option
        return None

    def _resolve_location_scope_value(
        self,
        value: str,
        pending: PendingClarification,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        if not pending.location_anchor_text:
            raise ClarificationStateMismatchError(
                "location scope has no anchor text"
            )
        base_state = replace(state, pending_clarification=None)
        if value == "reference_entity":
            reference = (
                replace(
                    state.reference,
                    location_hint_text=pending.location_anchor_text,
                )
                if state.reference
                else None
            )
            base_state = replace(
                base_state,
                explicit_target_location=None,
                reference=reference,
            )
            resolved = self.parse(
                message=self._category_message(state.target_category),
                state=base_state,
                has_user_location=has_user_location,
            )
            return replace(
                resolved,
                state_patch=replace(
                    resolved.state_patch,
                    clear_explicit_target_location=(
                        state.explicit_target_location is not None
                    ),
                    clear_pending_clarification=True,
                    reference=reference,
                ),
            )
        if value != "target_results":
            raise ClarificationStateMismatchError(
                "location scope option is not supported"
            )

        radius = (
            f" a {pending.radius_meters} metros"
            if pending.radius_meters is not None
            else ""
        )
        resolved = self.parse(
            message=(
                f"{self._category_message(state.target_category)} cerca de "
                f"{pending.location_anchor_text}{radius}"
            ),
            state=base_state,
            has_user_location=has_user_location,
        )
        return self._with_cleared_pending(resolved)

    @staticmethod
    def _with_cleared_pending(
        intent: ParsedPlaceChatIntent,
    ) -> ParsedPlaceChatIntent:
        return replace(
            intent,
            state_patch=replace(
                intent.state_patch,
                clear_pending_clarification=True,
            ),
        )

    def _category_message(self, category: str | None) -> str:
        definition = self._taxonomy.category(category)
        if definition and definition.aliases:
            return definition.aliases[0]
        return category or ""

    def _resolve_open_category_choice(
        self,
        value: str,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        category = " ".join(value.split()).strip()
        if not category:
            raise ClarificationStateMismatchError(
                "clarification category must not be empty"
            )
        hard_filters = _persistent_hard_filters(state.hard_filters)
        if state.city and "city" not in hard_filters:
            hard_filters["city"] = state.city
        if state.state and "state" not in hard_filters:
            hard_filters["state"] = state.state
        location, _ = self._location_intent(
            location_text=None,
            radius_meters=None,
            state=state,
            has_user_location=has_user_location,
        )
        semantic_parts = [category, *state.soft_preferences]
        if state.reference and state.reference.entity:
            semantic_parts.append(state.reference.entity)
        category_values = _ordered_unique(
            (category, *self._activity_category_values(category))
        )
        return ParsedPlaceChatIntent(
            action="recommendations",
            target_category=category,
            category_values=category_values,
            hard_filters=hard_filters,
            soft_preferences=state.soft_preferences,
            exclusions=state.exclusions,
            reference=state.reference,
            location=location,
            semantic_query=" ".join(_ordered_unique(semantic_parts)).strip(),
            confidence=1.0,
            state_patch=ConversationStatePatch(
                target_category=category,
                hard_filters=(
                    hard_filters if hard_filters != state.hard_filters else None
                ),
                clear_pending_clarification=True,
                taxonomy_version=self._taxonomy.version,
            ),
            category_evidence_terms=(category,),
            category_source="explicit",
            raw_category_phrase=category,
            intent_model_version=(
                f"{_DETERMINISTIC_INTENT_VERSION}+dynamic-clarification-v1"
            ),
        )

    def _location_intent(
        self,
        location_text: str | None,
        radius_meters: int | None,
        state: ConversationState,
        has_user_location: bool,
    ) -> tuple[LocationIntent, ExplicitTargetLocation | None]:
        if (
            location_text
            and prepare_for_embedding(location_text) in _CURRENT_LOCATION_VALUES
        ):
            return (
                LocationIntent(
                    scope="user_current_location",
                    source="user_current",
                    radius_meters=radius_meters,
                    strict_radius=radius_meters is not None,
                ),
                None,
            )
        if location_text:
            explicit = ExplicitTargetLocation(
                anchor_text=location_text,
                radius_meters=radius_meters,
                strict_radius=radius_meters is not None,
            )
            return (
                LocationIntent(
                    scope="target_results",
                    source="current_message",
                    anchor_text=location_text,
                    radius_meters=radius_meters,
                    strict_radius=radius_meters is not None,
                ),
                explicit,
            )
        if state.explicit_target_location:
            stored = state.explicit_target_location
            return (
                LocationIntent(
                    scope="target_results",
                    source="conversation_state",
                    anchor_text=stored.anchor_text,
                    resolved_place_id=stored.place_id,
                    radius_meters=stored.radius_meters,
                    strict_radius=stored.strict_radius,
                ),
                None,
            )
        if has_user_location:
            return (
                LocationIntent(
                    scope="user_current_location",
                    source="user_current",
                    radius_meters=radius_meters,
                    strict_radius=radius_meters is not None,
                ),
                None,
            )
        return (LocationIntent(scope="unresolved", source="none"), None)

    def _contextual_signals(self, message: str) -> _ContextualSignals:
        extractor = self._contextual_extractor
        if extractor is None:
            return _ContextualSignals(frame=None)

        try:
            frame = extractor.extract(message)
        except BertIntentExtractionError as exc:
            configured_version = getattr(
                extractor,
                "model_version",
                "unspecified",
            )
            _LOGGER.warning(
                "BERT place-intent extraction failed; using deterministic "
                "fallback (model_version=%s): %s",
                configured_version,
                exc,
            )
            return _ContextualSignals(
                frame=None,
                intent_model_version=(
                    f"{_DETERMINISTIC_INTENT_VERSION}+"
                    f"bert-fallback:{configured_version}"
                ),
            )

        category_spans = tuple(
            span
            for span in frame.by_type("CATEGORY")
            if span.polarity != "negative"
        )
        positive_preferences = tuple(
            span
            for span in frame.by_type("PREFERENCE")
            if span.polarity == "positive"
        )
        negative_spans = tuple(
            span
            for span in frame.spans
            if span.slot_type == "EXCLUSION" or span.polarity == "negative"
        )
        locations = frame.by_type("LOCATION")
        references = frame.by_type("REFERENCE")
        radii = frame.by_type("RADIUS")
        version = (
            f"bert-token:{frame.model_name}@{frame.model_version}+"
            f"{_DETERMINISTIC_INTENT_VERSION}"
        )
        return _ContextualSignals(
            frame=frame,
            category_phrases=_span_values(category_spans),
            preferences=_span_values(positive_preferences),
            exclusions=_span_values(negative_spans),
            location=_first_span_value(locations),
            reference=_first_span_value(references),
            radius_meters=(
                _radius_meters_from_text(radii[0].text) if radii else None
            ),
            category_confidence=(
                max(span.confidence for span in category_spans)
                if category_spans
                else None
            ),
            intent_model_version=version,
        )

    def _contextual_categories(
        self,
        contextual: _ContextualSignals,
        target_clause: str,
    ) -> tuple[str, ...]:
        if not contextual.category_phrases:
            return (
                self._matched_categories(target_clause)
                if contextual.frame is None
                else ()
            )

        categories: list[str] = []
        for phrase in contextual.category_phrases:
            if self._has_open_activity_preference(
                prepare_for_embedding(phrase)
            ):
                continue
            aligned = (
                self._activity_classifier.classify(phrase)
                if self._activity_classifier is not None
                else None
            )
            # Semantic alignment is optional and abstaining. Unknown values
            # pass through unchanged; no lexical alias table is consulted once
            # the contextual model has supplied a category span.
            categories.append(aligned.category if aligned is not None else phrase)
        return _ordered_unique(categories)

    def _activity_category_values(
        self,
        target_category: str | None,
    ) -> tuple[str, ...]:
        """Rehydrate an open catalog concept on conversation continuations."""

        if target_category is None or self._activity_classifier is None:
            return ()
        get_concept = getattr(self._activity_classifier, "get_concept", None)
        if callable(get_concept):
            concept = get_concept(target_category)
            if concept is not None:
                return tuple(getattr(concept, "storage_values", ()) or ())
        for concept in getattr(self._activity_classifier, "concepts", ()):
            if getattr(concept, "id", None) == target_category:
                return tuple(getattr(concept, "storage_values", ()) or ())
        return ()

    def _matched_categories(self, normalized: str) -> tuple[str, ...]:
        return _ordered_unique(
            category.canonical
            for category in self._taxonomy.categories
            if any(
                _contains_phrase(normalized, alias)
                for alias in (category.canonical, *category.aliases)
            )
        )

    @staticmethod
    def _category_query_term(
        normalized: str,
        category: CategoryDefinition | None,
    ) -> str:
        if category is None:
            return ""
        terms = category.evidence_terms or category.aliases
        for term in sorted(terms, key=len, reverse=True):
            if _contains_phrase(normalized, term):
                return term
        if terms:
            return terms[0]
        return category.canonical

    def _inferred_activity_category(
        self,
        normalized: str,
        *,
        allow_lexical_defaults: bool = True,
    ) -> PlaceCategoryInference | None:
        # Prefer the injected semantic/open-vocabulary model.  Lexical activity
        # rules remain a compatibility fallback during rollout, never a gate.
        if self._has_open_activity_preference(normalized):
            return None
        if self._activity_classifier is not None:
            inferred = self._activity_classifier.classify(normalized)
            if inferred is not None:
                return inferred
        if allow_lexical_defaults:
            for category, patterns in _ACTIVITY_CATEGORY_DEFAULTS:
                if _contains_any(normalized, patterns):
                    return PlaceCategoryInference(
                        category=category,
                        confidence=0.88,
                        source="lexical_activity",
                    )
        return None

    def _ranked_activity_alternatives(
        self,
        normalized: str,
    ) -> tuple[IntentAlternative, ...]:
        if self._has_open_activity_preference(normalized):
            return ()
        if self._activity_classifier is None:
            return ()
        rank_supported = getattr(
            self._activity_classifier,
            "rank_supported",
            None,
        )
        if not callable(rank_supported):
            return ()
        matches = rank_supported(normalized, limit=5)
        return tuple(
            IntentAlternative(
                key=str(match.concept_id),
                description=self._category_display_label(
                    str(match.concept_id),
                    str(match.label),
                ),
                confidence=max(0.0, min(1.0, (float(match.score) + 1.0) / 2.0)),
                category_values=tuple(
                    str(value)
                    for value in getattr(match, "storage_values", ())
                    if str(value).strip()
                ),
            )
            for match in matches
            if getattr(match, "concept_id", None)
        )

    def _category_display_label(self, category: str, fallback: str) -> str:
        definition = self._taxonomy.category(category)
        if definition is not None and definition.aliases:
            return _humanize_category_label(definition.aliases[0])
        return _humanize_category_label(fallback or category)

    @staticmethod
    def _target_clause(normalized: str) -> str:
        target_clause = _REFERENCE_PATTERN.sub(" ", normalized)
        target_clause = _LOCATION_PATTERN.sub(" ", target_clause)
        target_clause = _RADIUS_PATTERN.sub(" ", target_clause)
        return _EXCLUSION_PATTERN.sub(" ", target_clause)

    @staticmethod
    def _has_explicit_pending_choice(normalized: str) -> bool:
        return _contains_any(
            normalized,
            (
                "primera",
                "segunda",
                "opcion uno",
                "opcion dos",
                "otra zona",
                "solo para identificar",
                "es la referencia",
                "esa zona",
                "ese lugar",
            ),
        )

    def _matched_preferences(self, normalized: str) -> tuple[str, ...]:
        matched: list[str] = []
        for preference in self._taxonomy.preferences:
            if any(
                _contains_phrase(normalized, alias)
                for alias in preference.aliases
            ):
                matched.append(preference.canonical)
                matched.extend(preference.implies)
        return _ordered_unique(matched)

    @staticmethod
    def _extract_reference(normalized: str) -> str | None:
        match = _REFERENCE_PATTERN.search(normalized)
        return match.group("value").strip(" ,.;") if match else None

    @staticmethod
    def _extract_location(normalized: str) -> str | None:
        match = _LOCATION_PATTERN.search(normalized)
        return match.group("value").strip(" ,.;") if match else None

    @staticmethod
    def _extract_radius_meters(normalized: str) -> int | None:
        match = _RADIUS_PATTERN.search(normalized)
        if not match:
            return None
        value = float(match.group("value").replace(",", "."))
        if match.group("unit").startswith(("km", "kilometro")):
            value *= 1000
        return max(1, min(50_000, int(round(value))))

    def _clarification(
        self,
        pending: PendingClarification,
        unresolved: tuple[str, ...],
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        pending = ensure_legacy_pending_options(pending)
        clarification = to_public_clarification(pending)
        hard_filters = _persistent_hard_filters(state.hard_filters)
        location = (
            LocationIntent(scope="user_current_location", source="user_current")
            if has_user_location
            else LocationIntent(scope="unresolved", source="none")
        )
        return ParsedPlaceChatIntent(
            action="clarification",
            target_category=state.target_category,
            category_values=(),
            hard_filters=hard_filters,
            soft_preferences=state.soft_preferences,
            exclusions=state.exclusions,
            reference=state.reference,
            location=location,
            semantic_query="",
            confidence=0.5,
            state_patch=ConversationStatePatch(
                hard_filters=(
                    hard_filters if hard_filters != state.hard_filters else None
                ),
                pending_clarification=pending,
                taxonomy_version=self._taxonomy.version,
            ),
            category_source=(
                "conversation_state" if state.target_category else "unresolved"
            ),
            clarification=clarification,
            alternatives=tuple(
                IntentAlternative(
                    key=option.option_id,
                    description=option.label,
                    confidence=0.5,
                )
                for option in pending.options
            ),
            unresolved=unresolved,
            clarification_message=clarification.prompt,
        )


@lru_cache
def load_place_chat_taxonomy() -> PlaceChatTaxonomy:
    path = Path(__file__).with_name("place_chat_taxonomy.json")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return PlaceChatTaxonomy(
        version=str(payload["version"]),
        categories=tuple(
            CategoryDefinition(
                canonical=str(item["canonical"]),
                aliases=tuple(
                    prepare_for_embedding(str(alias))
                    for alias in item.get("aliases", [])
                ),
                storage_values=tuple(str(value) for value in item["storage_values"]),
                compatible_storage_values=tuple(
                    str(value)
                    for value in item.get("compatible_storage_values", [])
                ),
                evidence_terms=tuple(
                    prepare_for_embedding(str(value))
                    for value in item.get("evidence_terms", item.get("aliases", []))
                ),
            )
            for item in payload["categories"]
        ),
        preferences=tuple(
            PreferenceDefinition(
                canonical=str(item["canonical"]),
                aliases=tuple(
                    prepare_for_embedding(str(alias))
                    for alias in item.get("aliases", [])
                ),
                implies=tuple(str(value) for value in item.get("implies", [])),
            )
            for item in payload["preferences"]
        ),
    )


def _social_turn(normalized: str) -> tuple[str, str] | None:
    standalone = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    if standalone in _SOCIAL_GREETING_VALUES:
        return (
            "greeting",
            "¡Hola! Cuéntame qué tipo de lugar o actividad buscas y te ayudo "
            "a encontrar opciones cerca de ti.",
        )
    if standalone in _SOCIAL_THANKS_VALUES:
        return (
            "thanks",
            "¡Con gusto! Cuando quieras, dime qué lugar o plan buscas y seguimos.",
        )
    if standalone in _SOCIAL_FAREWELL_VALUES:
        return (
            "farewell",
            "¡Hasta luego! Cuando necesites ideas de lugares, aquí estaré.",
        )
    if standalone in _SOCIAL_ACKNOWLEDGEMENT_VALUES:
        return (
            "acknowledgement",
            "Perfecto. Dime qué lugar o actividad te interesa para continuar.",
        )
    return None


def _persistent_hard_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in filters.items()
        if key not in _TRANSIENT_HARD_FILTER_KEYS
    }


def _humanize_category_label(value: str) -> str:
    label = " ".join(value.replace("_", " ").split()).strip()
    if not label:
        return "Lugar"
    return label[:1].upper() + label[1:]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _remove_contextual_spans(
    target_clause: str,
    frame: IntentFrame | None,
) -> str:
    if frame is None:
        return target_clause
    cleaned = target_clause
    for span in frame.spans:
        if (
            span.slot_type not in {"EXCLUSION", "LOCATION", "REFERENCE", "RADIUS"}
            and span.polarity != "negative"
        ):
            continue
        normalized_value = prepare_for_embedding(span.text)
        if normalized_value:
            cleaned = re.sub(
                rf"(?<!\w){re.escape(normalized_value)}(?!\w)",
                " ",
                cleaned,
            )
    return " ".join(cleaned.split())


def _span_values(spans: Iterable[IntentSpan]) -> tuple[str, ...]:
    return _ordered_unique(
        " ".join(span.text.split()).strip(" ,.;") for span in spans
    )


def _first_span_value(spans: tuple[IntentSpan, ...]) -> str | None:
    values = _span_values(spans[:1])
    return values[0] if values else None


def _radius_meters_from_text(value: str) -> int | None:
    match = _RADIUS_VALUE_PATTERN.search(prepare_for_embedding(value))
    if match is None:
        return None
    distance = float(match.group("value").replace(",", "."))
    if match.group("unit").startswith(("km", "kilometro")):
        distance *= 1000
    return max(1, min(50_000, int(round(distance))))


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(
        _contains_phrase(text, prepare_for_embedding(value))
        for value in values
    )


def _last_affirmed_phrase_position(
    text: str,
    values: tuple[str, ...],
) -> int:
    position = -1
    for value in values:
        normalized = prepare_for_embedding(value)
        if not normalized:
            continue
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
        )
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 12) : match.start()]
            if re.search(r"\bno(?:\s+(?:la|el))?\s*$", prefix):
                continue
            position = max(position, match.start())
    return position


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
