from fastapi import APIRouter, Depends, Query

from app.modules.places.api.dependencies import (
    get_chat_places_use_case,
    get_evaluate_place_search_use_case,
    get_recommend_places_use_case,
    get_search_places_use_case,
)
from app.modules.places.api.schemas import (
    PlaceChatRequest,
    PlaceChatResponse,
    PlaceRecommendationRequest,
    PlaceRecommendationResponse,
    PlaceSearchMetricsResponse,
    PlaceSearchRequest,
    PlaceSearchResponse,
    engine_metrics_to_schema,
    place_to_schema,
    search_metrics_result_to_schema,
)
from app.modules.places.application.use_cases.chat_places import ChatPlacesUseCase
from app.modules.places.application.use_cases.evaluate_place_search import (
    EvaluatePlaceSearchUseCase,
)
from app.modules.places.application.use_cases.recommend_places import RecommendPlacesUseCase
from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.shared.security.rate_limit import rate_limit_placeholder

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
        evaluation_metrics=search_metrics_result_to_schema(result.evaluation_metrics),
        metadata=result.metadata,
    )


@router.post("/chat", response_model=PlaceChatResponse)
async def chat_places(
    payload: PlaceChatRequest,
    use_case: ChatPlacesUseCase = Depends(get_chat_places_use_case),
) -> PlaceChatResponse:
    result = await use_case.execute(
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
