import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.modules.places.api.dependencies import (
    get_chat_place_recommendations_use_case,
)
from app.modules.places.api.internal_auth import require_nlp_service_token
from app.modules.places.api.internal_chat_schemas import (
    InternalPlaceChatRequest,
    InternalPlaceChatResponse,
    internal_chat_result_to_schema,
)
from app.modules.places.application.use_cases.chat_place_recommendations import (
    ChatPlaceRecommendationsUseCase,
)
from app.modules.places.domain.errors import ClarificationStateMismatchError
from app.shared.config.settings import get_settings
from app.shared.security.rate_limit import rate_limit_placeholder


router = APIRouter(
    prefix="/internal/places",
    tags=["internal-places"],
    dependencies=[
        Depends(rate_limit_placeholder),
        Depends(require_nlp_service_token),
    ],
)


@router.post("/chat", response_model=InternalPlaceChatResponse)
async def chat_place_recommendations(
    payload: InternalPlaceChatRequest,
    use_case: ChatPlaceRecommendationsUseCase = Depends(
        get_chat_place_recommendations_use_case
    ),
) -> InternalPlaceChatResponse:
    settings = get_settings()
    if not settings.places_chat_v2_enabled:
        raise HTTPException(status_code=404, detail="Places chat V2 is disabled")
    if (
        payload.state.taxonomy_version is not None
        and payload.state.taxonomy_version != settings.places_chat_taxonomy_version
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Conversation taxonomy version is incompatible with "
                f"{settings.places_chat_taxonomy_version}"
            ),
        )
    if payload.candidate_limit > settings.places_chat_candidate_limit:
        raise HTTPException(
            status_code=422,
            detail=(
                "candidate_limit exceeds the configured service maximum of "
                f"{settings.places_chat_candidate_limit}"
            ),
        )
    try:
        async with asyncio.timeout(settings.request_timeout_seconds):
            result = await use_case.execute(
                message=payload.message,
                state=payload.state.to_domain(),
                user_latitude=payload.user_location.lat,
                user_longitude=payload.user_location.lng,
                candidate_limit=payload.candidate_limit,
                result_limit=payload.result_limit,
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
            detail="Internal places chat timed out",
        ) from exc
    return internal_chat_result_to_schema(result)
