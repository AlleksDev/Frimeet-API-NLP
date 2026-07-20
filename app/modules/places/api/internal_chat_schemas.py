from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsResult,
)
from app.modules.places.domain.chat_intent import (
    ClarificationChoice,
    ConversationState,
    ExplicitTargetLocation,
    PendingClarification,
    PendingClarificationOption,
    PlaceReference,
)


class UserLocationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class PlaceReferenceStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str | None = Field(default=None, max_length=160)
    place_id: str | None = Field(default=None, max_length=100)
    attributes: list[str] = Field(default_factory=list, max_length=30)
    location_hint_text: str | None = Field(default=None, max_length=160)
    location_hint_place_id: str | None = Field(default=None, max_length=100)

    def to_domain(self) -> PlaceReference:
        return PlaceReference(
            entity=self.entity,
            place_id=self.place_id,
            attributes=tuple(self.attributes),
            location_hint_text=self.location_hint_text,
            location_hint_place_id=self.location_hint_place_id,
        )


class ExplicitTargetLocationStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_text: str = Field(..., min_length=1, max_length=160)
    place_id: str | None = Field(default=None, max_length=100)
    label: str | None = Field(default=None, max_length=160)
    radius_meters: int | None = Field(default=None, ge=1, le=50_000)
    strict_radius: bool = False

    def to_domain(self) -> ExplicitTargetLocation:
        return ExplicitTargetLocation(
            anchor_text=self.anchor_text,
            place_id=self.place_id,
            label=self.label,
            radius_meters=self.radius_meters,
            strict_radius=self.strict_radius,
        )


class PendingClarificationOptionStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    value: str = Field(..., min_length=1, max_length=500)
    label: str = Field(..., min_length=1, max_length=160)
    place_id: str | None = Field(default=None, max_length=100)
    attributes: list[str] = Field(default_factory=list, max_length=30)

    def to_domain(self) -> PendingClarificationOption:
        return PendingClarificationOption(
            option_id=self.id,
            value=self.value,
            label=self.label,
            place_id=self.place_id,
            attributes=tuple(self.attributes),
        )


class PendingClarificationStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    kind: Literal[
        "target_category",
        "intent_category",
        "location_scope",
        "location_anchor",
        "reference_entity",
        "reference_location_anchor",
    ]
    options: list[PendingClarificationOptionStateSchema] = Field(
        default_factory=list,
        max_length=5,
    )
    location_anchor_text: str | None = Field(default=None, max_length=160)
    radius_meters: int | None = Field(default=None, ge=1, le=50_000)
    strict_radius: bool = False

    def to_domain(self) -> PendingClarification:
        return PendingClarification(
            clarification_id=str(self.id or uuid4()),
            kind=self.kind,
            options=tuple(option.to_domain() for option in self.options),
            location_anchor_text=self.location_anchor_text,
            radius_meters=self.radius_meters,
            strict_radius=self.strict_radius,
        )


class ConversationStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_category: str | None = Field(default=None, max_length=500)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: list[str] = Field(default_factory=list, max_length=30)
    exclusions: list[str] = Field(default_factory=list, max_length=30)
    reference: PlaceReferenceStateSchema | None = None
    explicit_target_location: ExplicitTargetLocationStateSchema | None = None
    pending_clarification: PendingClarificationStateSchema | None = None
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    taxonomy_version: str | None = Field(default=None, max_length=64)

    def to_domain(self) -> ConversationState:
        return ConversationState(
            target_category=self.target_category,
            hard_filters=dict(self.hard_filters),
            soft_preferences=tuple(self.soft_preferences),
            exclusions=tuple(self.exclusions),
            reference=self.reference.to_domain() if self.reference else None,
            explicit_target_location=(
                self.explicit_target_location.to_domain()
                if self.explicit_target_location
                else None
            ),
            pending_clarification=(
                self.pending_clarification.to_domain()
                if self.pending_clarification
                else None
            ),
            city=self.city,
            state=self.state,
            taxonomy_version=self.taxonomy_version,
        )


class ClarificationChoiceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarification_id: UUID
    option_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )

    def to_domain(self) -> ClarificationChoice:
        return ClarificationChoice(
            clarification_id=str(self.clarification_id),
            option_id=self.option_id,
        )


class InternalPlaceChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    turn: int = Field(..., ge=1)
    message: str = Field(..., min_length=1, max_length=1000)
    clarification_choice: ClarificationChoiceSchema | None = None
    state: ConversationStateSchema = Field(default_factory=ConversationStateSchema)
    user_location: UserLocationSchema
    candidate_limit: int = Field(default=30, ge=1, le=40)
    result_limit: int = Field(default=5, ge=1, le=8)

    @model_validator(mode="after")
    def validate_limits(self) -> "InternalPlaceChatRequest":
        if self.result_limit > self.candidate_limit:
            raise ValueError("result_limit cannot exceed candidate_limit")
        return self


class PlaceChatLocationDirectiveSchema(BaseModel):
    source: Literal["user_current", "explicit_anchor", "state_anchor", "unresolved"]
    scope: Literal[
        "target_results",
        "reference_entity",
        "user_current_location",
        "unresolved",
    ]
    anchor_place_id: str | None = None
    anchor_text: str | None = None
    radius_meters: int | None = None
    strict_radius: bool = False


class PlaceChatCandidateSchema(BaseModel):
    place_id: str
    content_score: float = Field(..., ge=0, le=1)
    semantic_score: float = Field(..., ge=0, le=1)
    lexical_score: float = Field(..., ge=0, le=1)
    match_level: Literal["exact", "family", "broad"]
    matched_reasons: list[str]


class ClarificationOptionSchema(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(..., min_length=1, max_length=80)
    message: str = Field(..., min_length=1, max_length=200)


class ClarificationSchema(BaseModel):
    id: UUID
    kind: Literal[
        "target_category",
        "intent_category",
        "location_scope",
        "location_anchor",
        "reference_entity",
        "reference_location_anchor",
    ]
    prompt: str = Field(..., min_length=1, max_length=500)
    options: list[ClarificationOptionSchema] = Field(..., min_length=2, max_length=5)


class InternalPlaceChatResponse(BaseModel):
    action: Literal["recommendations", "clarification", "no_match"]
    message: str
    state_patch: dict[str, Any]
    location_directive: PlaceChatLocationDirectiveSchema
    candidates: list[PlaceChatCandidateSchema]
    clarification: ClarificationSchema | None = None
    unresolved: list[str]
    intent_confidence: float = Field(..., ge=0, le=1)
    ranking_version: str
    taxonomy_version: str
    trace_id: str
    uncertainty: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


def internal_chat_result_to_schema(
    result: ChatPlaceRecommendationsResult,
) -> InternalPlaceChatResponse:
    directive = result.location_directive
    hypotheses = list(result.category_hypotheses)
    margin = (
        max(
            0.0,
            float(hypotheses[0]["probability"])
            - float(hypotheses[1]["probability"]),
        )
        if len(hypotheses) >= 2
        else None
    )
    return InternalPlaceChatResponse(
        action=result.action,
        message=result.message,
        state_patch=result.state_patch,
        location_directive=PlaceChatLocationDirectiveSchema(
            source=directive.source,
            scope=directive.scope,
            anchor_place_id=directive.anchor_place_id,
            anchor_text=directive.anchor_text,
            radius_meters=directive.radius_meters,
            strict_radius=directive.strict_radius,
        ),
        candidates=[
            PlaceChatCandidateSchema(
                place_id=candidate.place_id,
                content_score=round(candidate.content_score, 6),
                semantic_score=round(candidate.semantic_score, 6),
                lexical_score=round(candidate.lexical_score, 6),
                match_level=candidate.match_level,
                matched_reasons=list(candidate.matched_reasons),
            )
            for candidate in result.candidates
        ],
        clarification=(
            ClarificationSchema(
                id=UUID(result.clarification.clarification_id),
                kind=result.clarification.kind,
                prompt=result.clarification.prompt,
                options=[
                    ClarificationOptionSchema(
                        id=option.option_id,
                        label=option.label,
                        message=option.message,
                    )
                    for option in result.clarification.options
                ],
            )
            if result.clarification
            else None
        ),
        unresolved=list(result.unresolved),
        intent_confidence=round(result.intent_confidence, 6),
        ranking_version=result.ranking_version,
        taxonomy_version=result.taxonomy_version,
        trace_id=result.trace_id,
        uncertainty={
            "decision": (
                "review"
                if result.action == "recommendations" and result.unresolved
                else {
                    "recommendations": "auto",
                    "clarification": "clarify",
                    "no_match": "abstain",
                }[result.action]
            ),
            "reason": (
                result.unresolved[0]
                if result.unresolved
                else (
                    "sufficient_evidence"
                    if result.action == "recommendations"
                    else "catalog_exhausted"
                )
            ),
            "top_probability": round(result.intent_confidence, 6),
            "category_hypotheses": hypotheses,
            "category_margin": round(margin, 6) if margin is not None else None,
            "calibration_version": "uncalibrated-shadow-v1",
        },
        metadata={
            "used_llm": result.used_llm,
            "guard_reason": result.guard_reason,
            "category_source": result.category_source,
            "raw_category_phrase": result.raw_category_phrase,
            "intent_model_version": result.intent_model_version,
            "input_kind": (
                "non_search"
                if "non_search_input" in result.unresolved
                else "place_search"
            ),
            "effective_radius_meters": directive.radius_meters,
            "radius_is_strict": directive.strict_radius,
        },
    )
