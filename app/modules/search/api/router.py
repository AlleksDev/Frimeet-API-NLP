import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from app.modules.search.api.dependencies import get_search_all_use_case
from app.modules.search.api.cursor import decode_search_cursor
from app.modules.search.api.schemas import (
    GlobalSearchRequest,
    GlobalSearchResponse,
    result_to_schema,
)
from app.modules.search.application.use_cases.search_all import SearchAllUseCase
from app.modules.search.domain.models import ALL_SEARCH_RESOURCE_TYPES
from app.shared.config.settings import get_settings
from app.shared.security.rate_limit import rate_limit_placeholder

router = APIRouter(
    tags=["search"],
    dependencies=[Depends(rate_limit_placeholder)],
)


@router.post("/search", response_model=GlobalSearchResponse)
async def search_all(
    payload: GlobalSearchRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    use_case: SearchAllUseCase = Depends(get_search_all_use_case),
) -> GlobalSearchResponse:
    if payload.requester_id:
        expected_token = get_settings().search_internal_token
        bearer_token = _extract_bearer_token(authorization)
        if not expected_token or not bearer_token or not secrets.compare_digest(
            bearer_token, expected_token
        ):
            raise HTTPException(
                status_code=403,
                detail="A trusted internal token is required for requester-scoped search",
            )
    resource_types = (
        tuple(dict.fromkeys(payload.resource_types))
        if payload.resource_types
        else ALL_SEARCH_RESOURCE_TYPES
    )
    try:
        offsets = {
            resource_type: decode_search_cursor(
                cursor=cursor,
                resource_type=resource_type,
                query=payload.query,
            )
            for resource_type, cursor in payload.cursors.items()
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = await use_case.execute(
        query=payload.query,
        resource_types=resource_types,
        per_type_limit=payload.per_type_limit,
        top_limit=payload.top_limit,
        requester_id=str(payload.requester_id) if payload.requester_id else None,
        offsets=offsets,
    )
    return result_to_schema(result)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()
