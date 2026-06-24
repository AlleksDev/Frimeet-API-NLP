from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.nlp.llm.base import LLMProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard
from app.shared.tracing import new_response_id, new_trace_id


@dataclass(frozen=True)
class ChatPlacesResult:
    response_id: str
    nlp_trace_id: str
    message: str
    places: list[PlaceCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatPlacesUseCase:
    def __init__(
        self,
        search_use_case: SearchPlacesUseCase,
        llm_provider: LLMProvider,
        output_guard: PlaceChatOutputGuard,
    ) -> None:
        self._search_use_case = search_use_case
        self._llm_provider = llm_provider
        self._output_guard = output_guard

    async def execute(
        self,
        message: str,
        filters: PlaceFilters,
        limit: int = 5,
    ) -> ChatPlacesResult:
        response_id = new_response_id()
        trace_id = new_trace_id()
        search_result = await self._search_use_case.execute(
            query=message,
            filters=filters,
            limit=limit,
        )
        places = search_result.places
        context_places = [place.to_llm_context() for place in places]

        llm_provider = self._llm_provider.provider_name
        llm_model = self._llm_provider.model_name
        used_llm = False
        guard_reason = None

        try:
            llm_result = await self._llm_provider.generate_place_chat_response(
                user_intent=search_result.normalized_query,
                region=filters.city or filters.state,
                places=context_places,
            )
            llm_provider = llm_result.provider
            llm_model = llm_result.model
            guarded = self._output_guard.validate(
                message=llm_result.message,
                allowed_place_names=[place.name for place in places],
            )
            used_llm = not guarded.used_fallback
            guard_reason = guarded.reason
            final_message = guarded.message
        except Exception as exc:
            guarded = self._output_guard.fallback(reason=exc.__class__.__name__)
            final_message = guarded.message
            guard_reason = guarded.reason

        return ChatPlacesResult(
            response_id=response_id,
            nlp_trace_id=trace_id,
            message=final_message,
            places=places,
            metadata={
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "used_llm": used_llm,
                "guard_reason": guard_reason,
                "places_used_as_context": [place.id for place in places],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
