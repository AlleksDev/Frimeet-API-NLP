import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.modules.search.api.cursor import (
    build_search_cursor_context,
    decode_search_cursor,
)
from app.modules.search.api.dependencies import get_search_candidates_use_case
from app.modules.search.api.internal_auth import require_search_service_token
from app.modules.search.api.internal_schemas import (
    InternalSearchCandidatesRequest,
    InternalSearchCandidatesResponse,
    candidates_result_to_schema,
)
from app.modules.search.application.use_cases.search_candidates import (
    SearchCandidatesUseCase,
)
from app.shared.config.settings import get_settings
from app.shared.security.rate_limit import rate_limit_placeholder

router = APIRouter(
    prefix="/internal/search",
    tags=["internal-search"],
    dependencies=[
        Depends(rate_limit_placeholder),
        Depends(require_search_service_token),
    ],
)


@router.post(
    "/candidates",
    response_model=InternalSearchCandidatesResponse,
)
async def search_candidates(
    payload: InternalSearchCandidatesRequest,
    use_case: SearchCandidatesUseCase = Depends(get_search_candidates_use_case),
) -> InternalSearchCandidatesResponse:
    settings = get_settings()
    cursor_context = build_search_cursor_context(
        payload.cursor_context_payload(
            nearby_boost=settings.global_search_nearby_boost,
            policy_version=settings.global_search_threshold_policy_version,
        )
    )
    try:
        offsets = {
            resource_type: decode_search_cursor(
                cursor=cursor,
                resource_type=resource_type,
                context_fingerprint=cursor_context,
            )
            for resource_type, cursor in payload.cursors.items()
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        async with asyncio.timeout(settings.request_timeout_seconds):
            result = await use_case.execute(
                query=payload.query,
                resource_types=tuple(payload.resource_types),
                candidate_limit_per_type=payload.candidate_limit_per_type,
                as_of=payload.as_of,
                offsets=offsets,
                criteria=payload.to_domain_criteria(settings.global_search_nearby_boost),
                cursor_context=cursor_context,
            )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="Internal search timed out",
        ) from exc
    return candidates_result_to_schema(result)
