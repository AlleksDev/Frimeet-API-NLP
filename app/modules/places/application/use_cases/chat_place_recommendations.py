import asyncio
from dataclasses import dataclass, replace
import logging
import math
from typing import Any, Sequence

from app.modules.places.application.ports.chat_retriever import (
    HybridPlaceChatRetriever,
)
from app.modules.places.application.ports.intent_parser import PlaceChatIntentParser
from app.modules.places.application.ports.nearby_place_provider import (
    NearbyPlaceProvider,
)
from app.modules.places.application.ports.place_anchor_resolver import (
    PlaceAnchorResolver,
)
from app.modules.places.domain.clarifications import (
    category_display_label,
    new_anchor_clarification,
    new_category_clarification,
    to_public_clarification,
)
from app.modules.places.domain.chat_intent import (
    ChatAction,
    Clarification,
    ClarificationChoice,
    ConversationState,
    ExplicitTargetLocation,
    LocationIntent,
    ParsedPlaceChatIntent,
    PendingClarification,
    PlaceChatCandidate,
    PlaceChatLocationDirective,
    PlaceReference,
    ResolvedPlaceAnchor,
)
from app.shared.nlp.llm.base import LLMProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard
from app.shared.tracing import new_trace_id


logger = logging.getLogger(__name__)

_NON_CATEGORY_ALTERNATIVE_KEYS = {
    "location_scope",
    "reference_entity",
    "target_results",
    "user_current_location",
}


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
    clarification: Clarification | None = None
    category_source: str = "unresolved"
    used_llm: bool = False
    guard_reason: str | None = None
    category_hypotheses: tuple[dict[str, Any], ...] = ()
    raw_category_phrase: str | None = None
    intent_model_version: str = "deterministic-open-v3"


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
        minimum_hypothesis_confidence: float = 0.60,
        maximum_hypothesis_gap: float = 0.15,
        nearby_place_provider: NearbyPlaceProvider | None = None,
        default_radius_meters: int = 5_000,
        maximum_auto_radius_meters: int = 50_000,
    ) -> None:
        if not 0.0 <= minimum_intent_confidence <= 1.0:
            raise ValueError("minimum_intent_confidence must be between zero and one")
        if not 0.0 <= minimum_hypothesis_confidence <= 1.0:
            raise ValueError(
                "minimum_hypothesis_confidence must be between zero and one"
            )
        if not 0.0 <= maximum_hypothesis_gap <= 1.0:
            raise ValueError("maximum_hypothesis_gap must be between zero and one")
        if not 1 <= default_radius_meters <= 50_000:
            raise ValueError("default_radius_meters must be between 1 and 50000")
        if not default_radius_meters <= maximum_auto_radius_meters <= 50_000:
            raise ValueError(
                "maximum_auto_radius_meters must be between the default radius "
                "and 50000"
            )
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
        self._minimum_hypothesis_confidence = minimum_hypothesis_confidence
        self._maximum_hypothesis_gap = maximum_hypothesis_gap
        self._nearby_place_provider = nearby_place_provider
        self._default_radius_meters = default_radius_meters
        self._maximum_auto_radius_meters = maximum_auto_radius_meters

    async def execute(
        self,
        message: str,
        state: ConversationState,
        user_latitude: float | None,
        user_longitude: float | None,
        candidate_limit: int,
        result_limit: int,
        clarification_choice: ClarificationChoice | None = None,
    ) -> ChatPlaceRecommendationsResult:
        if (user_latitude is None) != (user_longitude is None):
            raise ValueError("user_latitude and user_longitude must be provided together")
        has_user_location = user_latitude is not None
        trace_id = new_trace_id()
        # Both the open-vocabulary classifier and the optional BERT extractor
        # are CPU-bound and load lazily. Keep their first inference off the
        # async request loop.
        intent = await asyncio.to_thread(
            self._intent_parser.parse,
            message,
            state,
            has_user_location,
            clarification_choice,
        )
        if intent.action == "no_match":
            return self._result(
                action="no_match",
                message=(
                    intent.response_message
                    or "Cuéntame qué tipo de lugar o actividad buscas."
                ),
                intent=intent,
                directive=self._location_directive(intent),
                candidates=(),
                unresolved=intent.unresolved,
                trace_id=trace_id,
            )
        if intent.action == "clarification":
            return self._clarification_result(intent, trace_id)

        resolved = await self._resolve_entities(intent, state)
        if isinstance(resolved, ChatPlaceRecommendationsResult):
            final_result = replace(resolved, trace_id=trace_id)
            self._log_result(final_result, intent)
            return final_result
        intent = resolved

        directive = self._location_directive(intent)
        if directive.source == "unresolved":
            region = state.city or state.state
            if not region:
                return self._result(
                    action="no_match",
                    message="No pude identificar una ubicacion util para esta busqueda.",
                    intent=intent,
                    directive=directive,
                    candidates=(),
                    unresolved=("location",),
                    trace_id=trace_id,
                )
            directive = PlaceChatLocationDirective(
                source="state_anchor",
                scope="target_results",
                anchor_text=region,
            )

        geographic_latitude = (
            user_latitude
            if directive.source == "user_current"
            else intent.location.latitude
        )
        geographic_longitude = (
            user_longitude
            if directive.source == "user_current"
            else intent.location.longitude
        )
        can_auto_expand_radius = False
        if (
            directive.source in {"user_current", "explicit_anchor", "state_anchor"}
            and self._nearby_place_provider is not None
            and geographic_latitude is not None
            and geographic_longitude is not None
        ):
            allows_auto_expansion = (
                directive.radius_meters is None and not directive.strict_radius
            )
            effective_radius = (
                directive.radius_meters or self._default_radius_meters
            )
            nearby_ids = await self._nearby_place_provider.get_nearby_place_ids(
                latitude=geographic_latitude,
                longitude=geographic_longitude,
                radius_meters=effective_radius,
            )
            if (
                not nearby_ids
                and allows_auto_expansion
                and effective_radius < self._maximum_auto_radius_meters
            ):
                effective_radius = self._maximum_auto_radius_meters
                nearby_ids = await self._nearby_place_provider.get_nearby_place_ids(
                    latitude=geographic_latitude,
                    longitude=geographic_longitude,
                    radius_meters=effective_radius,
                )
            directive = replace(directive, radius_meters=effective_radius)
            if not nearby_ids:
                return self._result(
                    action="no_match",
                    message=(
                        "No encontré lugares disponibles cerca de tu ubicación "
                        f"dentro de {self._distance_label(effective_radius)}. "
                        "Puedes indicar otra zona o intentarlo con un plan distinto."
                    ),
                    intent=intent,
                    directive=directive,
                    candidates=(),
                    unresolved=("nearby_catalog_empty",),
                    trace_id=trace_id,
                )
            intent = replace(
                intent,
                hard_filters={
                    **intent.hard_filters,
                    "place_ids": tuple(sorted(nearby_ids)),
                },
            )
            can_auto_expand_radius = (
                allows_auto_expansion
                and effective_radius < self._maximum_auto_radius_meters
            )

        if intent.confidence < self._minimum_intent_confidence:
            evidence_candidates = tuple(
                await self._retriever.retrieve(
                    intent=intent,
                    limit=min(candidate_limit, 12),
                )
            )
            pending = (
                self._category_clarification_from_hypotheses(
                    intent,
                    evidence_candidates,
                )
            )
            if pending is not None:
                clarified = self._with_pending_clarification(
                    intent,
                    pending,
                    unresolved=("intent_confidence",),
                )
                return self._clarification_result(
                    clarified,
                    trace_id,
                    directive=directive,
                )

            sufficient_candidates = tuple(
                candidate
                for candidate in evidence_candidates
                if candidate.metadata.get("retrieval_diagnostics", {}).get(
                    "meets_minimum_content_score"
                )
            )
            if evidence_candidates:
                review_candidates = (
                    sufficient_candidates or evidence_candidates
                )
                final_message, used_llm, guard_reason = await self._compose_message(
                    intent=intent,
                    candidates=review_candidates[:result_limit],
                    state=state,
                )
                return self._result(
                    action="recommendations",
                    message=final_message,
                    intent=intent,
                    directive=directive,
                    candidates=review_candidates,
                    unresolved=tuple(
                        dict.fromkeys(
                            (
                                *intent.unresolved,
                                "intent_confidence",
                                *(
                                    ()
                                    if sufficient_candidates
                                    else ("retrieval_evidence",)
                                ),
                            )
                        )
                    ),
                    trace_id=trace_id,
                    used_llm=used_llm,
                    guard_reason=guard_reason,
                )
            return self._result(
                action="no_match",
                message=self._uncertain_intent_message(intent, evidence_candidates),
                intent=intent,
                directive=directive,
                candidates=(),
                unresolved=("intent_confidence",),
                trace_id=trace_id,
            )

        candidates = tuple(
            await self._retriever.retrieve(intent=intent, limit=candidate_limit)
        )
        category_supported = self._has_category_supported_candidate(
            intent,
            candidates,
        )
        if (
            can_auto_expand_radius
            and (not candidates or (intent.target_category and not category_supported))
            and self._nearby_place_provider is not None
            and geographic_latitude is not None
            and geographic_longitude is not None
        ):
            expanded_radius = self._maximum_auto_radius_meters
            expanded_ids = await self._nearby_place_provider.get_nearby_place_ids(
                latitude=geographic_latitude,
                longitude=geographic_longitude,
                radius_meters=expanded_radius,
            )
            directive = replace(directive, radius_meters=expanded_radius)
            if expanded_ids:
                expanded_intent = replace(
                    intent,
                    hard_filters={
                        **intent.hard_filters,
                        "place_ids": tuple(sorted(expanded_ids)),
                    },
                )
                expanded_candidates = tuple(
                    await self._retriever.retrieve(
                        intent=expanded_intent,
                        limit=candidate_limit,
                    )
                )
                expanded_category_supported = (
                    self._has_category_supported_candidate(
                        expanded_intent,
                        expanded_candidates,
                    )
                )
                if (
                    not candidates
                    or expanded_category_supported
                    or not intent.target_category
                ):
                    intent = expanded_intent
                    candidates = expanded_candidates
                    category_supported = expanded_category_supported

        if intent.target_category and candidates and not category_supported:
            category_label = self._display_category(intent.target_category)
            radius_label = (
                f" dentro de {self._distance_label(directive.radius_meters)}"
                if directive.radius_meters is not None
                else ""
            )
            return self._result(
                action="no_match",
                message=(
                    f"No encontré opciones disponibles de {category_label}"
                    f"{radius_label}. Prueba con otra categoría o indícame "
                    "una zona diferente."
                ),
                intent=intent,
                directive=directive,
                candidates=(),
                unresolved=("category_availability",),
                trace_id=trace_id,
            )
        if not candidates:
            return self._result(
                action="no_match",
                message=(
                    "No encontré opciones suficientemente relacionadas con lo que "
                    "buscas. Puedes cambiar alguna preferencia o pedirme otro tipo "
                    "de lugar."
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
        has_sufficient_evidence = any(
            self._meets_content_threshold(candidate) for candidate in candidates
        )
        return self._result(
            action="recommendations",
            message=final_message,
            intent=intent,
            directive=directive,
            candidates=candidates,
            unresolved=(
                () if has_sufficient_evidence else ("retrieval_evidence",)
            ),
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
                    action="no_match",
                    message=(
                        f"No pude ubicar {location.anchor_text.title()}. "
                        "Intenta con otro punto de referencia."
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
                    unresolved=(),
                    trace_id="",
                )
            if self._anchors_are_ambiguous(anchors):
                clarified = self._with_pending_clarification(
                    intent,
                    new_anchor_clarification(
                        "location_anchor",
                        location.anchor_text,
                        anchors,
                    ),
                    unresolved=("location_anchor",),
                )
                return self._clarification_result(clarified, "")
            anchor = anchors[0]
            resolved_location = replace(
                location,
                resolved_place_id=anchor.place_id,
                latitude=anchor.latitude,
                longitude=anchor.longitude,
            )
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
        elif (
            location.anchor_text
            and location.resolved_place_id
            and (location.latitude is None or location.longitude is None)
        ):
            anchors = list(
                await self._anchor_resolver.resolve(
                    text=location.anchor_text,
                    city=state.city,
                    state=state.state,
                    limit=5,
                )
            )
            selected = next(
                (
                    anchor
                    for anchor in anchors
                    if anchor.place_id == location.resolved_place_id
                ),
                None,
            )
            if selected is not None:
                resolved_intent = replace(
                    resolved_intent,
                    location=replace(
                        location,
                        latitude=selected.latitude,
                        longitude=selected.longitude,
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
                        action="no_match",
                        message=(
                            "No pude ubicar el lugar que identifica tu referencia. "
                            "Intenta con otra referencia."
                        ),
                        intent=resolved_intent,
                        directive=self._location_directive(resolved_intent),
                        candidates=(),
                        unresolved=(),
                        trace_id="",
                    )
                selected_location = (
                    next(
                        (
                            anchor
                            for anchor in location_anchors
                            if anchor.place_id == reference.location_hint_place_id
                        ),
                        None,
                    )
                    if reference.location_hint_place_id
                    else None
                )
                if reference.location_hint_place_id and selected_location is None:
                    return self._result(
                        action="no_match",
                        message="La ubicacion seleccionada ya no esta disponible.",
                        intent=resolved_intent,
                        directive=self._location_directive(resolved_intent),
                        candidates=(),
                        unresolved=(),
                        trace_id="",
                    )
                if selected_location is not None:
                    reference_location = selected_location
                elif self._anchors_are_ambiguous(location_anchors):
                    clarified = self._with_pending_clarification(
                        resolved_intent,
                        new_anchor_clarification(
                            "reference_location_anchor",
                            reference.location_hint_text,
                            location_anchors,
                        ),
                        unresolved=("reference_location_anchor",),
                    )
                    return self._clarification_result(clarified, "")
                else:
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
                clarified = self._with_pending_clarification(
                    resolved_intent,
                    new_anchor_clarification(
                        "reference_entity",
                        reference.entity,
                        anchors,
                    ),
                    unresolved=("reference_entity",),
                )
                return self._clarification_result(clarified, "")
            if anchor:
                resolved_reference = PlaceReference(
                    entity=reference.entity,
                    place_id=anchor.place_id,
                    attributes=tuple(
                        dict.fromkeys((*reference.attributes, *anchor.attributes))
                    ),
                    location_hint_text=reference.location_hint_text,
                    location_hint_place_id=reference.location_hint_place_id,
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

    def _category_clarification_from_hypotheses(
        self,
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
    ) -> PendingClarification | None:
        scores: dict[str, float] = {}
        labels: dict[str, str] = {}
        category_values: dict[str, tuple[str, ...]] = {}
        if intent.target_category:
            scores[intent.target_category] = intent.confidence
            category_values[intent.target_category] = tuple(
                dict.fromkeys(
                    (
                        intent.target_category,
                        *intent.category_values,
                    )
                )
            )

        for alternative in intent.alternatives:
            key = alternative.key.strip()
            if not key or key in _NON_CATEGORY_ALTERNATIVE_KEYS:
                continue
            scores[key] = max(scores.get(key, 0.0), alternative.confidence)
            category_values[key] = tuple(
                dict.fromkeys((key, *alternative.category_values))
            )
            if alternative.description.strip():
                labels[key] = alternative.description.strip()

        # Prefer the source catalog's localized label over an English/static
        # classifier label whenever a supporting candidate exposes one.
        for category, values in category_values.items():
            localized_labels = tuple(
                str(candidate.metadata.get("category_label") or "").strip()
                for candidate in candidates
                if self._candidate_supports_category_values(candidate, values)
                and str(candidate.metadata.get("category_label") or "").strip()
            )
            if localized_labels:
                labels[category] = localized_labels[0]

        ranked = sorted(
            (
                (category, score)
                for category, score in scores.items()
                if any(
                    self._candidate_supports_category_values(
                        candidate,
                        category_values.get(category, (category,)),
                    )
                    for candidate in candidates
                )
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked or ranked[0][1] < self._minimum_hypothesis_confidence:
            return None
        top_score = ranked[0][1]
        ordered = tuple(
            category
            for category, score in ranked
            if score >= self._minimum_hypothesis_confidence
            and top_score - score <= self._maximum_hypothesis_gap
        )[:5]
        if len(ordered) < 2:
            return None
        return new_category_clarification(
            ordered,
            kind="intent_category",
            labels=labels,
        )

    @staticmethod
    def _uncertain_intent_message(
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
    ) -> str:
        categories = tuple(
            dict.fromkeys(
                str(candidate.metadata.get("category_label") or "").strip()
                for candidate in candidates
                if str(candidate.metadata.get("category_label") or "").strip()
            )
        )[:3]
        if len(categories) > 1:
            evidence = ", ".join(categories[:-1]) + f" y {categories[-1]}"
            return (
                f"Encontre senales relacionadas con {evidence}, pero no pude "
                "determinar con suficiente confianza cual describe tu plan. "
                "Cuentame que actividad quieres hacer."
            )
        if categories:
            return (
                f"La busqueda apunta a {categories[0]}, pero la intencion sigue "
                "siendo ambigua. Cuentame que actividad quieres hacer para afinarla."
            )
        if intent.soft_preferences:
            preferences = ", ".join(
                preference.replace("_", " ")
                for preference in intent.soft_preferences[:3]
            )
            return (
                f"Entendi que buscas algo {preferences}, pero no pude determinar "
                "con suficiente confianza el tipo de lugar. Describe la actividad "
                "que tienes en mente."
            )
        return (
            "No pude determinar con suficiente confianza el tipo de lugar. "
            "Describe la actividad o el plan que tienes en mente."
        )

    async def _compose_message(
        self,
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
        state: ConversationState,
    ) -> tuple[str, bool, str | None]:
        has_sufficient_evidence = any(
            self._meets_content_threshold(candidate) for candidate in candidates
        )
        response_mode = (
            "low_confidence"
            if (
                intent.confidence < self._minimum_intent_confidence
                or not has_sufficient_evidence
            )
            else "confident"
        )
        fallback = self._template_message(intent, candidates, response_mode)
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
                response_mode=response_mode,
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
                response_mode=response_mode,
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
        response_mode: str = "confident",
    ) -> str:
        if response_mode == "low_confidence":
            return (
                "Encontre opciones semanticamente relacionadas, aunque la evidencia "
                "todavia es debil. Revisalas como sugerencias y ajusta tu busqueda "
                "si no representan el plan que tienes en mente."
            )
        exact = sum(candidate.match_level == "exact" for candidate in candidates)
        family = sum(candidate.match_level == "family" for candidate in candidates)
        if exact:
            return "Encontre opciones con coincidencias directas para el estilo que buscas."
        if family:
            return "Encontre opciones relacionadas con el tema y las preferencias que mencionaste."
        category = (
            category_display_label(intent.target_category).casefold()
            if intent.target_category
            else "lugar"
        )
        return f"Encontre opciones de {category} que pueden encajar con tu solicitud."

    @staticmethod
    def _meets_content_threshold(candidate: PlaceChatCandidate) -> bool:
        diagnostics = candidate.metadata.get("retrieval_diagnostics", {})
        if "meets_minimum_content_score" in diagnostics:
            return bool(diagnostics["meets_minimum_content_score"])
        return candidate.content_score > 0.0

    @classmethod
    def _candidate_supports_category_values(
        cls,
        candidate: PlaceChatCandidate,
        values: Sequence[str],
    ) -> bool:
        if not cls._meets_content_threshold(candidate):
            return False
        supported = {
            normalized
            for value in values
            if (normalized := _normalized_category(value))
        }
        return _candidate_has_category_evidence(candidate, supported)

    @staticmethod
    def _has_category_supported_candidate(
        intent: ParsedPlaceChatIntent,
        candidates: Sequence[PlaceChatCandidate],
    ) -> bool:
        if not intent.target_category:
            return True

        requested_values = {
            normalized
            for value in (
                intent.target_category,
                *intent.category_values,
                *intent.compatible_category_values,
            )
            if (normalized := _normalized_category(value))
        }
        for candidate in candidates:
            diagnostics = candidate.metadata.get("retrieval_diagnostics", {})
            category_match = diagnostics.get("category_match")
            if category_match not in {None, "none", "not_requested"}:
                return True
            if _candidate_has_category_evidence(candidate, requested_values):
                return True
        return False

    @staticmethod
    def _display_category(value: str) -> str:
        return _display_category(value).casefold()

    @staticmethod
    def _distance_label(radius_meters: int) -> str:
        if radius_meters >= 1_000 and radius_meters % 1_000 == 0:
            return f"{radius_meters // 1_000} km"
        return f"{radius_meters} m"

    @staticmethod
    def _candidate_context(candidate: PlaceChatCandidate) -> dict[str, Any]:
        return {
            "id": candidate.place_id,
            "name": candidate.name,
            "category": candidate.category,
            "score": round(candidate.content_score, 4),
            "tags": candidate.metadata.get("tags"),
            "short_description": candidate.metadata.get("short_description"),
            "category_label": candidate.metadata.get("category_label"),
            "attribute_states": candidate.metadata.get("attribute_states"),
            "attribute_terms": candidate.metadata.get("attribute_terms"),
            "entertainment_features": candidate.metadata.get(
                "entertainment_features"
            ),
            "contained_items": candidate.metadata.get("contained_items"),
            "menu_items": candidate.metadata.get("menu_items"),
            "matched_reasons": list(candidate.matched_reasons),
            "match_level": candidate.match_level,
        }

    def _clarification_result(
        self,
        intent: ParsedPlaceChatIntent,
        trace_id: str,
        *,
        directive: PlaceChatLocationDirective | None = None,
    ) -> ChatPlaceRecommendationsResult:
        if intent.clarification is None:
            raise RuntimeError("clarification action requires structured options")
        return self._result(
            action="clarification",
            message=intent.clarification.prompt,
            intent=intent,
            directive=directive or self._location_directive(intent),
            candidates=(),
            unresolved=intent.unresolved,
            trace_id=trace_id,
        )

    @staticmethod
    def _with_pending_clarification(
        intent: ParsedPlaceChatIntent,
        pending: PendingClarification,
        unresolved: tuple[str, ...],
    ) -> ParsedPlaceChatIntent:
        clarification = to_public_clarification(pending)
        patch = intent.state_patch
        source_hard_filters = (
            patch.hard_filters
            if patch.hard_filters is not None
            else intent.hard_filters
        )
        persistent_hard_filters = _persistent_hard_filters(source_hard_filters)
        return replace(
            intent,
            action="clarification",
            confidence=min(intent.confidence, 0.75),
            state_patch=replace(
                patch,
                target_category=(
                    patch.target_category or intent.target_category
                ),
                hard_filters=(
                    persistent_hard_filters
                    if patch.hard_filters is not None or persistent_hard_filters
                    else None
                ),
                soft_preferences=(
                    patch.soft_preferences
                    if patch.soft_preferences is not None
                    else (intent.soft_preferences or None)
                ),
                exclusions=(
                    patch.exclusions
                    if patch.exclusions is not None
                    else (intent.exclusions or None)
                ),
                pending_clarification=pending,
            ),
            clarification=clarification,
            unresolved=unresolved,
            clarification_message=clarification.prompt,
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
            clarification=(
                intent.clarification if action == "clarification" else None
            ),
            category_source=intent.category_source,
            used_llm=used_llm,
            guard_reason=guard_reason,
            category_hypotheses=tuple(
                {
                    "id": alternative.key,
                    "label": alternative.description,
                    "probability": round(alternative.confidence, 6),
                }
                for alternative in intent.alternatives
                if alternative.key not in _NON_CATEGORY_ALTERNATIVE_KEYS
            )[:5],
            raw_category_phrase=intent.raw_category_phrase,
            intent_model_version=intent.intent_model_version,
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


def _persistent_hard_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}
    return {
        key: value
        for key, value in filters.items()
        if key != "place_ids"
    }


def _normalized_category(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("_", " ").replace("-", " ").casefold().split())


def _display_category(value: str) -> str:
    return category_display_label(value)


def _candidate_category_values(candidate: PlaceChatCandidate) -> set[str]:
    values: set[str] = set()
    for raw_value in (
        candidate.category,
        candidate.metadata.get("category_label"),
        candidate.metadata.get("tags"),
        candidate.metadata.get("tag_names"),
    ):
        for value in _text_values(raw_value):
            normalized = _normalized_category(value)
            if normalized:
                values.add(normalized)
    return values


def _candidate_has_category_evidence(
    candidate: PlaceChatCandidate,
    requested_values: set[str],
) -> bool:
    if not requested_values:
        return False
    if _candidate_category_values(candidate) & requested_values:
        return True

    raw_evidence: list[str] = []
    for raw_value in (
        candidate.category,
        candidate.metadata.get("category_label"),
        candidate.metadata.get("tags"),
        candidate.metadata.get("tag_names"),
    ):
        raw_evidence.extend(_text_values(raw_value))
    haystack = f" {' '.join(_normalized_category(value) for value in raw_evidence)} "
    return any(f" {value} " in haystack for value in requested_values)


def _text_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(
            part.strip()
            for part in value.replace(";", ",").split(",")
            if part.strip()
        )
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


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
