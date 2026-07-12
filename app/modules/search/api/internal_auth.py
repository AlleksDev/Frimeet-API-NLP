import hmac

from fastapi import Header, HTTPException, status

from app.shared.config.settings import get_settings


def require_search_service_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    expected = get_settings().post_feed_internal_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP_SERVICE_TOKEN no configurado",
        )
    scheme, separator, received = (authorization or "").partition(" ")
    if not separator or scheme.casefold() != "bearer" or not received.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if not hmac.compare_digest(received.strip(), expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
