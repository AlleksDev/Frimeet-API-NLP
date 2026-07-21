import asyncio
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Query

from app.modules.places.api.dependencies import (
    get_chat_places_use_case,
    get_chat_place_recommendations_use_case,
    get_evaluate_place_search_use_case,
    get_recommend_places_use_case,
    get_search_places_use_case,
)
from app.modules.places.api.internal_chat_schemas import (
    internal_chat_result_to_schema,
)
from app.modules.places.api.schemas import (
    PlaceChatRequest,
    PlaceChatResponse,
    PlaceRecommendationRequest,
    PlaceRecommendationResponse,
    PlaceResultSchema,
    PlaceSearchMetricsResponse,
    PlaceSearchRequest,
    PlaceSearchResponse,
    engine_metrics_to_schema,
    place_to_schema,
    search_metrics_result_to_schema,
)
from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsUseCase,
)
from app.modules.places.domain.chat_intent import ConversationState
from app.modules.places.domain.errors import ClarificationStateMismatchError
from app.modules.places.domain.taxonomy_version import (
    is_compatible_place_chat_taxonomy,
)
from app.modules.places.application.use_cases.evaluate_place_search import (
    EvaluatePlaceSearchUseCase,
)
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.shared.security.rate_limit import rate_limit_placeholder
from app.shared.config.settings import get_settings
from app.shared.tracing import new_response_id

router = APIRouter(
    prefix="/places",
    tags=["places"],
    dependencies=[Depends(rate_limit_placeholder)],
)


@router.get("/search/metrics", response_model=PlaceSearchMetricsResponse)
@router.post("/search/metrics", response_model=PlaceSearchMetricsResponse)
async def evaluate_place_search(
    k: int = Query(default=5, ge=1, le=20),
    use_case: EvaluatePlaceSearchUseCase = Depends(get_evaluate_place_search_use_case),
) -> PlaceSearchMetricsResponse:
    result = await use_case.execute(k=k)
    return search_metrics_result_to_schema(result)


@router.post("/search", response_model=PlaceSearchResponse)
async def search_places(
    payload: PlaceSearchRequest,
    use_case: SearchPlacesUseCase = Depends(get_search_places_use_case),
) -> PlaceSearchResponse:
    result = await use_case.execute(
        query=payload.query,
        filters=payload.to_domain_filters(),
        limit=payload.limit,
        latitude=payload.lat,
        longitude=payload.lng,
        radius_meters=payload.radius,
    )
    return PlaceSearchResponse(
        query=result.query,
        places=[place_to_schema(place) for place in result.places],
        metrics=engine_metrics_to_schema(result.metrics),
    )


@router.post("/recommendations", response_model=PlaceRecommendationResponse)
async def recommend_places(
    payload: PlaceRecommendationRequest,
    use_case: RecommendPlacesUseCase = Depends(get_recommend_places_use_case),
) -> PlaceRecommendationResponse:
    result = await use_case.execute(
        query=payload.query,
        filters=payload.to_domain_filters(),
        limit=payload.limit,
        latitude=payload.lat,
        longitude=payload.lng,
        radius_meters=payload.radius,
    )
    return PlaceRecommendationResponse(
        query=result.query,
        message=result.message,
        places=[place_to_schema(place) for place in result.places],
        metrics=engine_metrics_to_schema(result.metrics),
        metadata=result.metadata,
    )


@router.post(
    "/chat",
    response_model=PlaceChatResponse,
    response_model_exclude_none=True,
)
async def chat_places(
    payload: PlaceChatRequest,
    legacy_use_case: ChatPlacesUseCase = Depends(get_chat_places_use_case),
    semantic_use_case: ChatPlaceRecommendationsUseCase = Depends(
        get_chat_place_recommendations_use_case
    ),
) -> PlaceChatResponse:
    settings = get_settings()
    use_semantic_chat = settings.places_chat_v2_enabled and any(
        (
            payload.conversation_id is not None,
            payload.conversation_state is not None,
            payload.clarification_choice is not None,
            payload.user_location is not None,
        )
    )
    if use_semantic_chat:
        state = (
            payload.conversation_state.to_domain()
            if payload.conversation_state
            else ConversationState()
        )
        if not is_compatible_place_chat_taxonomy(
            state.taxonomy_version,
            settings.places_chat_taxonomy_version,
        ):
            raise HTTPException(
                status_code=409,
                detail="Conversation taxonomy version is incompatible",
            )
        request_filters = payload.to_domain_filters()
        hard_filters = dict(state.hard_filters)
        for key, value in {
            "city": request_filters.city,
            "state": request_filters.state,
            "price_range": request_filters.price_range,
            "occasion": request_filters.occasion,
        }.items():
            if value is not None:
                hard_filters[key] = value
        state = replace(
            state,
            target_category=state.target_category or request_filters.category,
            hard_filters=hard_filters,
            city=state.city or request_filters.city,
            state=state.state or request_filters.state,
        )
        location = payload.user_location
        candidate_limit = (
            payload.candidate_limit or settings.places_chat_candidate_limit
        )
        if candidate_limit > settings.places_chat_candidate_limit:
            raise HTTPException(
                status_code=422,
                detail=(
                    "candidate_limit exceeds the configured service maximum of "
                    f"{settings.places_chat_candidate_limit}"
                ),
            )
        try:
            async with asyncio.timeout(settings.request_timeout_seconds):
                semantic_result = await semantic_use_case.execute(
                    message=payload.message,
                    state=state,
                    user_latitude=location.lat if location else None,
                    user_longitude=location.lng if location else None,
                    candidate_limit=candidate_limit,
                    result_limit=payload.limit,
                    clarification_choice=(
                        payload.clarification_choice.to_domain()
                        if payload.clarification_choice
                        else None
                    ),
                )
        except ClarificationStateMismatchError as exc:
            raise HTTPException(
                status_code=409,
                detail="Clarification choice does not match the current state",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Places chat timed out",
            ) from exc
        structured = internal_chat_result_to_schema(semantic_result)
        decision = (
            "review"
            if semantic_result.action == "recommendations"
            and semantic_result.unresolved
            else {
                "recommendations": "auto",
                "clarification": "clarify",
                "no_match": "abstain",
            }[semantic_result.action]
        )
        reason = (
            semantic_result.unresolved[0]
            if semantic_result.unresolved
            else (
                "sufficient_evidence"
                if semantic_result.action == "recommendations"
                else "catalog_exhausted"
            )
        )
        return PlaceChatResponse(
            response_id=new_response_id(),
            nlp_trace_id=semantic_result.trace_id,
            action=semantic_result.action,
            message=semantic_result.message,
            places=[
                PlaceResultSchema(
                    id=candidate.place_id,
                    name=candidate.name,
                    score=round(candidate.content_score, 4),
                    category=candidate.category,
                    city=candidate.metadata.get("city"),
                    state=candidate.metadata.get("state"),
                    metadata={
                        **candidate.metadata,
                        "semantic_score": candidate.semantic_score,
                        "lexical_score": candidate.lexical_score,
                        "match_level": candidate.match_level,
                        "matched_reasons": list(candidate.matched_reasons),
                    },
                )
                for candidate in semantic_result.candidates[: payload.limit]
            ],
            state_patch=semantic_result.state_patch,
            location_directive=structured.location_directive,
            clarification=structured.clarification,
            unresolved=list(semantic_result.unresolved),
            intent_confidence=semantic_result.intent_confidence,
            ranking_version=semantic_result.ranking_version,
            taxonomy_version=semantic_result.taxonomy_version,
            uncertainty={
                **structured.uncertainty,
                "decision": decision,
                "reason": reason,
            },
            metadata={
                **structured.metadata,
                "pipeline": "places-chat-semantic-v2",
                "conversation_id": (
                    str(payload.conversation_id)
                    if payload.conversation_id is not None
                    else None
                ),
                "turn": payload.turn,
            },
        )

    result = await legacy_use_case.execute(
        message=payload.message,
        filters=payload.to_domain_filters(),
        limit=payload.limit,
    )
    return PlaceChatResponse(
        response_id=result.response_id,
        nlp_trace_id=result.nlp_trace_id,
        message=result.message,
        places=[place_to_schema(place) for place in result.places],
        metadata=result.metadata,
    )
