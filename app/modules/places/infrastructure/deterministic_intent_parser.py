from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable

from app.modules.places.application.ports.activity_classifier import (
    PlaceActivityClassifier,
)
from app.modules.places.domain.chat_intent import (
    CategoryInferenceSource,
    ConversationState,
    ConversationStatePatch,
    ExplicitTargetLocation,
    IntentAlternative,
    LocationIntent,
    PendingClarification,
    PlaceCategoryInference,
    ParsedPlaceChatIntent,
    PlaceReference,
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
    r"\s+(?:pero|aunque)\b|\s+y\s+(?:con|que|quiero|busco)\b|$)"
)
_CURRENT_LOCATION_VALUES = {"mi", "aqui", "donde estoy", "mi ubicacion"}
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


class DeterministicPlaceChatIntentParser:
    def __init__(
        self,
        taxonomy: PlaceChatTaxonomy | None = None,
        activity_classifier: PlaceActivityClassifier | None = None,
    ) -> None:
        self._taxonomy = taxonomy or load_place_chat_taxonomy()
        self._activity_classifier = activity_classifier

    def parse(
        self,
        message: str,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        normalized = prepare_for_embedding(message)
        target_clause = self._target_clause(normalized)
        if state.pending_clarification:
            categories = self._matched_categories(target_clause)
            inferred = (
                self._inferred_activity_category(target_clause)
                if not categories
                else None
            )
            if (
                (categories or inferred is not None)
                and not self._has_explicit_pending_choice(normalized)
            ):
                fresh_intent = self.parse(
                    message=message,
                    state=replace(state, pending_clarification=None),
                    has_user_location=has_user_location,
                )
                return replace(
                    fresh_intent,
                    state_patch=replace(
                        fresh_intent.state_patch,
                        clear_pending_clarification=True,
                    ),
                )
            return self._resolve_pending_clarification(
                normalized=normalized,
                state=state,
                has_user_location=has_user_location,
            )

        reference_text = self._extract_reference(normalized)
        location_text = self._extract_location(normalized)
        radius_meters = self._extract_radius_meters(normalized)
        excluded_texts = tuple(
            match.group("value").strip(" ,.;")
            for match in _EXCLUSION_PATTERN.finditer(normalized)
        )

        # Category words inside references, exclusions, or geographic anchors
        # do not describe the requested result type. In "cafeteria cerca del
        # parque", only cafe is a hard category.
        categories = self._matched_categories(target_clause)
        if len(categories) > 1:
            return self._clarification(
                message=(
                    "Mencionaste varios tipos de lugar. ¿Cual quieres que busque "
                    "primero: " + ", ".join(categories) + "?"
                ),
                unresolved=("target_category",),
                alternatives=tuple(
                    IntentAlternative(
                        key=category,
                        description=f"Buscar lugares de categoria {category}",
                        confidence=0.5,
                    )
                    for category in categories
                ),
                state=state,
                has_user_location=has_user_location,
            )

        explicit_category = categories[0] if categories else None
        inferred = (
            self._inferred_activity_category(target_clause)
            if explicit_category is None
            else None
        )
        inferred_category = inferred.category if inferred else None
        requested_category = explicit_category or inferred_category
        target_category = requested_category or state.target_category
        if explicit_category:
            category_source: CategoryInferenceSource = "explicit"
        elif inferred:
            category_source = inferred.source
        elif state.target_category:
            category_source = "conversation_state"
        else:
            category_source = "unresolved"
        category_changed = bool(
            requested_category
            and state.target_category
            and requested_category != state.target_category
        )

        if location_text:
            location_text = _RADIUS_PATTERN.sub("", location_text).strip(" ,.;")

        detected_preferences = self._matched_preferences(normalized)
        explicit_exclusions = _ordered_unique(
            exclusion
            for excluded_text in excluded_texts
            for exclusion in self._matched_preferences(excluded_text)
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
        hard_filters = dict(state.hard_filters)
        if "economico" in merged_preferences or _contains_any(
            target_clause,
            ("mas barato", "economico", "barato"),
        ):
            hard_filters["price_preference"] = "lower"

        if target_category is None:
            return self._clarification(
                message=(
                    "¿Que tipo de lugar buscas: una cafeteria, restaurante, "
                    "parque u otra opcion?"
                ),
                unresolved=("target_category",),
                alternatives=(),
                state=state,
                has_user_location=has_user_location,
            )

        category = self._taxonomy.category(target_category)
        category_values = category.storage_values if category else (target_category,)

        if (
            reference_text
            and location_text
            and location_text not in _CURRENT_LOCATION_VALUES
        ):
            return self._location_scope_clarification(
                target_category=target_category,
                category_values=category_values,
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
        semantic_parts = [target_category, semantic_target, *merged_preferences]
        if reference and reference.entity:
            semantic_parts.append(reference.entity)
        semantic_query = " ".join(_ordered_unique(semantic_parts)).strip()

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
            taxonomy_version=self._taxonomy.version,
        )
        if explicit_category:
            confidence = 0.96
        elif inferred_category:
            confidence = inferred.confidence if inferred else 0.88
        else:
            confidence = 0.84
        if reference_text:
            confidence -= 0.05
        return ParsedPlaceChatIntent(
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
            category_source=category_source,
        )

    def _location_scope_clarification(
        self,
        target_category: str,
        category_values: tuple[str, ...],
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
        pending = PendingClarification(
            kind="location_scope",
            location_anchor_text=location_text,
            radius_meters=radius_meters,
            strict_radius=radius_meters is not None,
        )
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
            category_source=category_source,
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
            clarification_message=(
                f"¿Quieres opciones cerca de {location_text.title()} o lugares "
                f"parecidos a {reference_text.title()} aunque esten en otra zona?"
            ),
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
                message=(
                    "¿Uso ese lugar para ubicar los resultados o solo para "
                    "identificar la referencia?"
                ),
                unresolved=("location_scope",),
                alternatives=(),
                state=state,
                has_user_location=has_user_location,
            )

        base_state = replace(state, pending_clarification=None)
        if reference_choice:
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
                message=state.target_category or "",
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

        radius = (
            f" a {pending.radius_meters} metros"
            if pending.radius_meters is not None
            else ""
        )
        resolved = self.parse(
            message=(
                f"{state.target_category or ''} cerca de "
                f"{pending.location_anchor_text}{radius}"
            ),
            state=base_state,
            has_user_location=has_user_location,
        )
        return replace(
            resolved,
            state_patch=replace(
                resolved.state_patch,
                clear_pending_clarification=True,
            ),
        )

    def _location_intent(
        self,
        location_text: str | None,
        radius_meters: int | None,
        state: ConversationState,
        has_user_location: bool,
    ) -> tuple[LocationIntent, ExplicitTargetLocation | None]:
        if location_text and location_text in _CURRENT_LOCATION_VALUES:
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
                ),
                None,
            )
        return (LocationIntent(scope="unresolved", source="none"), None)

    def _matched_categories(self, normalized: str) -> tuple[str, ...]:
        return _ordered_unique(
            category.canonical
            for category in self._taxonomy.categories
            if any(
                _contains_phrase(normalized, alias)
                for alias in category.aliases
            )
        )

    def _inferred_activity_category(
        self,
        normalized: str,
    ) -> PlaceCategoryInference | None:
        for category, patterns in _ACTIVITY_CATEGORY_DEFAULTS:
            if _contains_any(normalized, patterns):
                return PlaceCategoryInference(
                    category=category,
                    confidence=0.88,
                    source="lexical_activity",
                )
        if self._activity_classifier is None:
            return None
        inferred = self._activity_classifier.classify(normalized)
        if inferred is None or self._taxonomy.category(inferred.category) is None:
            return None
        return inferred

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
        message: str,
        unresolved: tuple[str, ...],
        alternatives: tuple[IntentAlternative, ...],
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        location = (
            LocationIntent(scope="user_current_location", source="user_current")
            if has_user_location
            else LocationIntent(scope="unresolved", source="none")
        )
        return ParsedPlaceChatIntent(
            action="clarification",
            target_category=state.target_category,
            category_values=(),
            hard_filters=dict(state.hard_filters),
            soft_preferences=state.soft_preferences,
            exclusions=state.exclusions,
            reference=state.reference,
            location=location,
            semantic_query="",
            confidence=0.5,
            state_patch=ConversationStatePatch(
                taxonomy_version=self._taxonomy.version
            ),
            category_source=(
                "conversation_state" if state.target_category else "unresolved"
            ),
            alternatives=alternatives,
            unresolved=unresolved,
            clarification_message=message,
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


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(
        _contains_phrase(text, prepare_for_embedding(value))
        for value in values
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
