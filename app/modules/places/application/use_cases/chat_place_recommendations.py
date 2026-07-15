from dataclasses import dataclass, replace
import logging
import math
from typing import Any, Sequence

from app.modules.places.application.ports.chat_retriever import (
    HybridPlaceChatRetriever,
)
from app.modules.places.application.ports.intent_parser import PlaceChatIntentParser
from app.modules.places.application.ports.place_anchor_resolver import (
    PlaceAnchorResolver,
)
from app.modules.places.domain.chat_intent import (
    ChatAction,
    ConversationState,
    ExplicitTargetLocation,
    LocationIntent,
    ParsedPlaceChatIntent,
    PlaceChatCandidate,
    PlaceChatLocationDirective,
    PlaceReference,
    ResolvedPlaceAnchor,
)
from app.shared.nlp.llm.base import LLMProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard
from app.shared.tracing import new_trace_id


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatPlaceRecommendationsResult:
    action: ChatAction
    message: str
    state_patch: dict[str, Any]
    location_directive: PlaceChatLocationDirective
    candidates: tuple[PlaceChatCandidate, ...]
    unresolved: tuple[str, ...]
    intent_confidence: float
    ranking_version: str
    taxonomy_version: str
    trace_id: str
    category_source: str = "unresolved"
    used_llm: bool = False
    guard_reason: str | None = None


class ChatPlaceRecommendationsUseCase:
    def __init__(
        self,
        intent_parser: PlaceChatIntentParser,
        anchor_resolver: PlaceAnchorResolver,
        retriever: HybridPlaceChatRetriever,
        llm_provider: LLMProvider,
        output_guard: PlaceChatOutputGuard,
        ranking_version: str,
        taxonomy_version: str,
        llm_enabled: bool = True,
        anchor_ambiguity_delta: float = 0.15,
        minimum_intent_confidence: float = 0.70,
    ) -> None:
        self._intent_parser = intent_parser
        self._anchor_resolver = anchor_resolver
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._output_guard = output_guard
        self._ranking_version = ranking_version
        self._taxonomy_version = taxonomy_version
        self._llm_enabled = llm_enabled
        self._anchor_ambiguity_delta = anchor_ambiguity_delta
        self._minimum_intent_confidence = minimum_intent_confidence

    async def execute(
        self,
        message: str,
        state: ConversationState,
        user_latitude: float,
        user_longitude: float,
        candidate_limit: int,
        result_limit: int,
    ) -> ChatPlaceRecommendationsResult:
        del user_latitude, user_longitude
        trace_id = new_trace_id()
        intent = self._intent_parser.parse(
            message=message,
            state=state,
            has_user_location=True,
        )
        if intent.action == "clarification":
            return self._clarification_result(intent, trace_id)
        if intent.confidence < self._minimum_intent_confidence:
            return self._result(
                action="clarification",
                message="No estoy seguro de haber entendido el tipo de lugar. ¿Puedes describirlo de otra forma?",
                intent=intent,
                directive=self._location_directive(intent),
                candidates=(),
                unresolved=("intent_confidence",),
                trace_id=trace_id,
            )

        resolved = await self._resolve_entities(intent, state)
        if isinstance(resolved, ChatPlaceRecommendationsResult):
            final_result = replace(resolved, trace_id=trace_id)
            self._log_result(final_result, intent)
            return final_result
        intent = resolved

        directive = self._location_directive(intent)
        if directive.source == "unresolved":
            return self._result(
                action="clarification",
                message="No pude identificar la ubicacion. ¿Puedes indicar el lugar o la zona con mas detalle?",
                intent=intent,
                directive=directive,
                candidates=(),
                unresolved=("location",),
                trace_id=trace_id,
            )

        candidates = tuple(
            await self._retriever.retrieve(intent=intent, limit=candidate_limit)
        )
        if not candidates:
            return self._result(
                action="no_match",
                message=(
                    "No encontre opciones suficientemente relacionadas con lo que buscas. "
                    "Puedes cambiar alguna preferencia o pedirme otro tipo de lugar."
                ),
                intent=intent,
                directive=directive,
                candidates=(),
                unresolved=(),
                trace_id=trace_id,
            )

        final_message, used_llm, guard_reason = await self._compose_message(
            intent=intent,
            candidates=candidates[:result_limit],
            state=state,
        )
        return self._result(
            action="recommendations",
            message=final_message,
            intent=intent,
            directive=directive,
            candidates=candidates,
            unresolved=(),
            trace_id=trace_id,
            used_llm=used_llm,
            guard_reason=guard_reason,
        )

    async def _resolve_entities(
        self,
        intent: ParsedPlaceChatIntent,
        state: ConversationState,
    ) -> ParsedPlaceChatIntent | ChatPlaceRecommendationsResult:
        resolved_intent = intent
        location = intent.location
        if location.anchor_text and not location.resolved_place_id:
            anchors = list(
                await self._anchor_resolver.resolve(
                    text=location.anchor_text,
                    city=state.city,
                    state=state.state,
                    limit=3,
                )
            )
            if not anchors:
                return self._result(
                    action="clarification",
                    message=(
                        f"No pude ubicar {location.anchor_text.title()}. "
                        "¿Puedes indicar la ciudad o elegir otro punto de referencia?"
                    ),
                    intent=replace(
                        intent,
                        state_patch=replace(
                            intent.state_patch,
                            explicit_target_location=None,
                        ),
                    ),
                    directive=PlaceChatLocationDirective(
                        source="unresolved",
                        scope="unresolved",
                        anchor_text=location.anchor_text,
                    ),
                    candidates=(),
                    unresolved=("location_anchor",),
                    trace_id="",
                )
            if self._anchors_are_ambiguous(anchors):
                names = ", ".join(anchor.name for anchor in anchors[:3])
                return self._result(
                    action="clarification",
                    message=f"Encontre varios lugares posibles: {names}. ¿A cual te refieres?",
                    intent=intent,
                    directive=PlaceChatLocationDirective(
                        source="unresolved",
                        scope="unresolved",
                        anchor_text=location.anchor_text,
                    ),
                    candidates=(),
                    unresolved=("location_anchor",),
                    trace_id="",
                )
            anchor = anchors[0]
            resolved_location = replace(location, resolved_place_id=anchor.place_id)
            explicit = ExplicitTargetLocation(
                anchor_text=location.anchor_text,
                place_id=anchor.place_id,
                label=anchor.name,
                radius_meters=location.radius_meters,
                strict_radius=location.strict_radius,
            )
            resolved_intent = replace(
                resolved_intent,
                location=resolved_location,
                state_patch=replace(
                    resolved_intent.state_patch,
                    explicit_target_location=explicit,
                ),
            )

        reference = resolved_intent.reference
        if reference and reference.entity and not reference.place_id:
            reference_location: ResolvedPlaceAnchor | None = None
            if reference.location_hint_text:
                location_anchors = list(
                    await self._anchor_resolver.resolve(
                        text=reference.location_hint_text,
                        city=state.city,
                        state=state.state,
                        limit=3,
                    )
                )
                if not location_anchors:
                    return self._result(
                        action="clarification",
                        message=(
                            "No pude ubicar el lugar que identifica tu referencia. "
                            "¿Puedes darme otro nombre o la ciudad?"
                        ),
                        intent=resolved_intent,
                        directive=self._location_directive(resolved_intent),
                        candidates=(),
                        unresolved=("reference_location_anchor",),
                        trace_id="",
                    )
                if self._anchors_are_ambiguous(location_anchors):
                    names = ", ".join(
                        anchor.name for anchor in location_anchors[:3]
                    )
                    return self._result(
                        action="clarification",
                        message=(
                            f"Encontre varios lugares para ubicar la referencia: "
                            f"{names}. ¿A cual te refieres?"
                        ),
                        intent=resolved_intent,
                        directive=self._location_directive(resolved_intent),
                        candidates=(),
                        unresolved=("reference_location_anchor",),
                        trace_id="",
                    )
                reference_location = location_anchors[0]

            anchors = list(
                await self._anchor_resolver.resolve(
                    text=reference.entity,
                    city=state.city,
                    state=state.state,
                    limit=3,
                )
            )
            anchor, is_ambiguous = self._select_reference_anchor(
                anchors,
                reference_location,
            )
            if is_ambiguous:
                names = ", ".join(anchor.name for anchor in anchors[:3])
                return self._result(
                    action="clarification",
                    message=f"Encontre varias referencias posibles: {names}. ¿Cual quieres usar como ejemplo?",
                    intent=resolved_intent,
                    directive=self._location_directive(resolved_intent),
                    candidates=(),
                    unresolved=("reference_entity",),
                    trace_id="",
                )
            if anchor:
                resolved_reference = PlaceReference(
                    entity=reference.entity,
                    place_id=anchor.place_id,
                    attributes=tuple(
                        dict.fromkeys((*reference.attributes, *anchor.attributes))
                    ),
                    location_hint_text=reference.location_hint_text,
                )
                resolved_intent = replace(
                    resolved_intent,
                    reference=resolved_reference,
                    semantic_query=" ".join(
                        dict.fromkeys(
                            (
                                resolved_intent.semantic_query,
                                *anchor.attributes,
                            )
                        )
                    ).strip(),
                    state_patch=replace(
                        resolved_intent.state_patch,
                        reference=resolved_reference,
                    ),
                )
        return resolved_intent

    def _anchors_are_ambiguous(self, anchors: Sequence[Any]) -> bool:
        return bool(
            len(anchors) > 1
            and abs(float(anchors[0].score) - float(anchors[1].score))
            <= self._anchor_ambiguity_delta
        )

    def _select_reference_anchor(
        self,
        anchors: Sequence[ResolvedPlaceAnchor],
        location_hint: ResolvedPlaceAnchor | None,
    ) -> tuple[ResolvedPlaceAnchor | None, bool]:
        if not anchors:
            return None, False
        if (
            location_hint
            and location_hint.latitude is not None
            and location_hint.longitude is not None
            and all(
                anchor.latitude is not None and anchor.longitude is not None
                for anchor in anchors
            )
        ):
            by_distance = sorted(
                [
                    (
                        _distance_meters(location_hint, anchor),
                        anchor,
                    )
                    for anchor in anchors
                ],
                key=lambda item: (item[0], item[1].place_id),
            )
            if len(by_distance) > 1 and by_distance[1][0] - by_distance[0][0] <= 100:
                return None, True
            return by_distance[0][1], False
        if self._anchors_are_ambiguous(anchors):
            return None, True
        return anchors[0], False

    @staticmethod
    def _location_directive(
        intent: ParsedPlaceChatIntent,
    ) -> PlaceChatLocationDirective:
        location = intent.location
        if location.source == "current_message" and location.resolved_place_id:
            source = "explicit_anchor"
        elif location.source == "conversation_state" and location.resolved_place_id:
            source = "state_anchor"
        elif location.source == "user_current":
            source = "user_current"
        else:
            source = "unresolved"
        return PlaceChatLocationDirective(
            source=source,
            scope=location.scope,
            anchor_place_id=location.resolved_place_id,
            anchor_text=location.anchor_text,
            radius_meters=location.radius_meters,
            strict_radius=location.strict_radius,
        )

    async def _compose_message(
        self,
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
        state: ConversationState,
    ) -> tuple[str, bool, str | None]:
        fallback = self._template_message(intent, candidates)
        if not self._llm_enabled:
            return fallback, False, "llm_disabled"
        try:
            result = await self._llm_provider.generate_place_chat_response(
                user_intent=(
                    intent.semantic_query
                    + ". Redacta de forma general y no menciones nombres propios."
                ),
                region=state.city or state.state,
                places=[self._candidate_context(candidate) for candidate in candidates],
                response_mode="confident",
            )
            if any(
                candidate.name.casefold() in result.message.casefold()
                for candidate in candidates
                if candidate.name
            ):
                return fallback, False, "candidate_name_deferred_to_main_api"
            guarded = self._output_guard.validate(
                message=result.message,
                allowed_place_names=[candidate.name for candidate in candidates],
            )
            if guarded.used_fallback:
                return fallback, False, guarded.reason
            return guarded.message, True, None
        except Exception as exc:
            return fallback, False, exc.__class__.__name__

    @staticmethod
    def _template_message(
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
    ) -> str:
        exact = sum(candidate.match_level == "exact" for candidate in candidates)
        family = sum(candidate.match_level == "family" for candidate in candidates)
        if exact:
            return "Encontre opciones con coincidencias directas para el estilo que buscas."
        if family:
            return "Encontre opciones relacionadas con el tema y las preferencias que mencionaste."
        category = intent.target_category or "lugar"
        return f"Encontre opciones de {category} que pueden encajar con tu solicitud."

    @staticmethod
    def _candidate_context(candidate: PlaceChatCandidate) -> dict[str, Any]:
        return {
            "id": candidate.place_id,
            "name": candidate.name,
            "category": candidate.category,
            "score": round(candidate.content_score, 4),
            "tags": candidate.metadata.get("tags"),
            "short_description": candidate.metadata.get("short_description"),
            "matched_reasons": list(candidate.matched_reasons),
            "match_level": candidate.match_level,
        }

    def _clarification_result(
        self,
        intent: ParsedPlaceChatIntent,
        trace_id: str,
    ) -> ChatPlaceRecommendationsResult:
        return self._result(
            action="clarification",
            message=intent.clarification_message or "Necesito un poco mas de informacion.",
            intent=intent,
            directive=self._location_directive(intent),
            candidates=(),
            unresolved=intent.unresolved,
            trace_id=trace_id,
        )

    def _result(
        self,
        action: ChatAction,
        message: str,
        intent: ParsedPlaceChatIntent,
        directive: PlaceChatLocationDirective,
        candidates: Sequence[PlaceChatCandidate],
        unresolved: tuple[str, ...],
        trace_id: str,
        used_llm: bool = False,
        guard_reason: str | None = None,
    ) -> ChatPlaceRecommendationsResult:
        result = ChatPlaceRecommendationsResult(
            action=action,
            message=message,
            state_patch=intent.state_patch.as_dict(),
            location_directive=directive,
            candidates=tuple(candidates),
            unresolved=unresolved,
            intent_confidence=intent.confidence,
            ranking_version=self._ranking_version,
            taxonomy_version=self._taxonomy_version,
            trace_id=trace_id,
            category_source=intent.category_source,
            used_llm=used_llm,
            guard_reason=guard_reason,
        )
        if trace_id:
            self._log_result(result, intent)
        return result

    @staticmethod
    def _log_result(
        result: ChatPlaceRecommendationsResult,
        intent: ParsedPlaceChatIntent,
    ) -> None:
        logger.info(
            "places_chat trace_id=%s action=%s category=%s category_source=%s "
            "confidence=%.3f "
            "location_source=%s location_scope=%s candidates=%d "
            "preferences=%d exclusions=%d hard_filter_keys=%s "
            "ranking_version=%s taxonomy_version=%s used_llm=%s guard=%s",
            result.trace_id,
            result.action,
            intent.target_category,
            result.category_source,
            result.intent_confidence,
            result.location_directive.source,
            result.location_directive.scope,
            len(result.candidates),
            len(intent.soft_preferences),
            len(intent.exclusions),
            ",".join(sorted(intent.hard_filters)),
            result.ranking_version,
            result.taxonomy_version,
            result.used_llm,
            result.guard_reason,
        )


def _distance_meters(
    left: ResolvedPlaceAnchor,
    right: ResolvedPlaceAnchor,
) -> float:
    if (
        left.latitude is None
        or left.longitude is None
        or right.latitude is None
        or right.longitude is None
    ):
        return math.inf
    left_latitude = math.radians(left.latitude)
    right_latitude = math.radians(right.latitude)
    latitude_delta = right_latitude - left_latitude
    longitude_delta = math.radians(right.longitude - left.longitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(left_latitude)
        * math.cos(right_latitude)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 6_371_000 * 2 * math.atan2(
        math.sqrt(haversine),
        math.sqrt(max(0.0, 1 - haversine)),
    )
