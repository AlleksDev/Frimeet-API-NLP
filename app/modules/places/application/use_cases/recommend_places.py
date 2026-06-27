from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_metrics import SearchEngineMetrics
from app.shared.nlp.llm.base import LLMProvider
from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard


@dataclass(frozen=True)
class RecommendPlacesResult:
    query: str
    message: str
    places: list[PlaceCandidate]
    metrics: SearchEngineMetrics
    metadata: dict[str, Any] = field(default_factory=dict)


class RecommendPlacesUseCase:
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
        query: str,
        filters: PlaceFilters,
        limit: int = 10,
    ) -> RecommendPlacesResult:
        search_result = await self._search_use_case.execute(
            query=query,
            filters=filters,
            limit=limit,
        )
        places = search_result.places
        llm_provider = self._llm_provider.provider_name
        llm_model = self._llm_provider.model_name
        used_llm = False
        guard_reason = None

        try:
            llm_result = await self._llm_provider.generate_place_chat_response(
                user_intent=search_result.normalized_query,
                region=filters.city or filters.state,
                places=[place.to_llm_context() for place in places],
            )
            llm_provider = llm_result.provider
            llm_model = llm_result.model
            guarded = self._output_guard.validate(
                message=llm_result.message,
                allowed_place_names=[place.name for place in places],
            )
            message = guarded.message
            used_llm = not guarded.used_fallback
            guard_reason = guarded.reason
        except Exception as exc:
            guarded = self._output_guard.fallback(reason=exc.__class__.__name__)
            message = guarded.message
            guard_reason = guarded.reason

        return RecommendPlacesResult(
            query=search_result.query,
            message=message,
            places=places,
            metrics=search_result.metrics,
            metadata={
                "strategy": "pgvector_candidates_plus_tfidf_ranking",
                "ranking": "tfidf_cosine",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "used_llm": used_llm,
                "guard_reason": guard_reason,
                "places_used_as_context": [place.id for place in places],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
