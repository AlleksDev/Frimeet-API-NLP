from dataclasses import dataclass, field
from typing import Any, Literal


ChatAction = Literal["recommendations", "clarification", "no_match"]
LocationScope = Literal[
    "target_results",
    "reference_entity",
    "user_current_location",
    "unresolved",
]
LocationSource = Literal[
    "current_message",
    "conversation_state",
    "user_current",
    "none",
]
MatchLevel = Literal["exact", "family", "broad"]
CategoryInferenceSource = Literal[
    "explicit",
    "lexical_activity",
    "semantic_activity",
    "conversation_state",
    "unresolved",
]
ClarificationKind = Literal[
    "target_category",
    "intent_category",
    "location_scope",
    "location_anchor",
    "reference_entity",
    "reference_location_anchor",
]


@dataclass(frozen=True)
class PlaceReference:
    entity: str | None = None
    place_id: str | None = None
    attributes: tuple[str, ...] = ()
    location_hint_text: str | None = None
    location_hint_place_id: str | None = None


@dataclass(frozen=True)
class ExplicitTargetLocation:
    anchor_text: str
    place_id: str | None = None
    label: str | None = None
    radius_meters: int | None = None
    strict_radius: bool = False


@dataclass(frozen=True)
class PendingClarificationOption:
    option_id: str
    value: str
    label: str
    place_id: str | None = None
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingClarification:
    kind: ClarificationKind
    clarification_id: str = ""
    options: tuple[PendingClarificationOption, ...] = ()
    location_anchor_text: str | None = None
    radius_meters: int | None = None
    strict_radius: bool = False


@dataclass(frozen=True)
class ClarificationOption:
    option_id: str
    label: str
    message: str


@dataclass(frozen=True)
class Clarification:
    clarification_id: str
    kind: ClarificationKind
    prompt: str
    options: tuple[ClarificationOption, ...]


@dataclass(frozen=True)
class ClarificationChoice:
    clarification_id: str
    option_id: str


@dataclass(frozen=True)
class ConversationState:
    target_category: str | None = None
    hard_filters: dict[str, Any] = field(default_factory=dict)
    soft_preferences: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    reference: PlaceReference | None = None
    explicit_target_location: ExplicitTargetLocation | None = None
    pending_clarification: PendingClarification | None = None
    city: str | None = None
    state: str | None = None
    taxonomy_version: str | None = None


@dataclass(frozen=True)
class LocationIntent:
    scope: LocationScope
    source: LocationSource
    anchor_text: str | None = None
    resolved_place_id: str | None = None
    radius_meters: int | None = None
    strict_radius: bool = False
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class IntentAlternative:
    key: str
    description: str
    confidence: float
    category_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlaceCategoryInference:
    category: str
    confidence: float
    source: Literal["lexical_activity", "semantic_activity"]
    category_values: tuple[str, ...] = ()
    label: str | None = None


@dataclass(frozen=True)
class ConversationStatePatch:
    target_category: str | None = None
    hard_filters: dict[str, Any] | None = None
    soft_preferences: tuple[str, ...] | None = None
    exclusions: tuple[str, ...] | None = None
    reference: PlaceReference | None = None
    explicit_target_location: ExplicitTargetLocation | None = None
    pending_clarification: PendingClarification | None = None
    clear_target_category: bool = False
    clear_reference: bool = False
    clear_explicit_target_location: bool = False
    clear_pending_clarification: bool = False
    taxonomy_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.target_category is not None:
            payload["target_category"] = self.target_category
        if self.clear_target_category:
            payload["target_category"] = None
        if self.hard_filters is not None:
            payload["hard_filters"] = self.hard_filters
        if self.soft_preferences is not None:
            payload["soft_preferences"] = list(self.soft_preferences)
        if self.exclusions is not None:
            payload["exclusions"] = list(self.exclusions)
        if self.reference is not None:
            payload["reference"] = {
                "entity": self.reference.entity,
                "place_id": self.reference.place_id,
                "attributes": list(self.reference.attributes),
                "location_hint_text": self.reference.location_hint_text,
                "location_hint_place_id": self.reference.location_hint_place_id,
            }
        if self.explicit_target_location is not None:
            payload["explicit_target_location"] = {
                "anchor_text": self.explicit_target_location.anchor_text,
                "place_id": self.explicit_target_location.place_id,
                "label": self.explicit_target_location.label,
                "radius_meters": self.explicit_target_location.radius_meters,
                "strict_radius": self.explicit_target_location.strict_radius,
            }
        if self.pending_clarification is not None:
            payload["pending_clarification"] = {
                "id": self.pending_clarification.clarification_id,
                "kind": self.pending_clarification.kind,
                "options": [
                    {
                        "id": option.option_id,
                        "value": option.value,
                        "label": option.label,
                        "place_id": option.place_id,
                        "attributes": list(option.attributes),
                    }
                    for option in self.pending_clarification.options
                ],
                "location_anchor_text": (
                    self.pending_clarification.location_anchor_text
                ),
                "radius_meters": self.pending_clarification.radius_meters,
                "strict_radius": self.pending_clarification.strict_radius,
            }
        if self.clear_reference:
            payload["reference"] = None
        if self.clear_explicit_target_location:
            payload["explicit_target_location"] = None
        if self.clear_pending_clarification:
            payload["pending_clarification"] = None
        if self.taxonomy_version is not None:
            payload["taxonomy_version"] = self.taxonomy_version
        return payload


@dataclass(frozen=True)
class ParsedPlaceChatIntent:
    action: ChatAction
    target_category: str | None
    category_values: tuple[str, ...]
    hard_filters: dict[str, Any]
    soft_preferences: tuple[str, ...]
    exclusions: tuple[str, ...]
    reference: PlaceReference | None
    location: LocationIntent
    semantic_query: str
    confidence: float
    state_patch: ConversationStatePatch
    compatible_category_values: tuple[str, ...] = ()
    category_evidence_terms: tuple[str, ...] = ()
    category_source: CategoryInferenceSource = "unresolved"
    clarification: Clarification | None = None
    alternatives: tuple[IntentAlternative, ...] = ()
    unresolved: tuple[str, ...] = ()
    clarification_message: str | None = None
    response_message: str | None = None
    raw_category_phrase: str | None = None
    intent_model_version: str = "deterministic-open-v3"


@dataclass(frozen=True)
class ResolvedPlaceAnchor:
    place_id: str
    name: str
    category: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    attributes: tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class PlaceChatCandidate:
    place_id: str
    name: str
    category: str | None
    content_score: float
    semantic_score: float
    lexical_score: float
    match_level: MatchLevel
    matched_reasons: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaceChatLocationDirective:
    source: Literal["user_current", "explicit_anchor", "state_anchor", "unresolved"]
    scope: LocationScope
    anchor_place_id: str | None = None
    anchor_text: str | None = None
    radius_meters: int | None = None
    strict_radius: bool = False
