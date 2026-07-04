import base64
import hashlib
import json
import secrets
from typing import Any

from app.modules.search.domain.models import SearchResourceType
CURSOR_VERSION = 1
MAX_SEARCH_OFFSET = 1000


def build_search_cursor_context(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def encode_search_cursor(
    resource_type: SearchResourceType,
    context_fingerprint: str,
    offset: int,
) -> str:
    payload = {
        "v": CURSOR_VERSION,
        "r": resource_type.value,
        "c": context_fingerprint,
        "o": offset,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii").rstrip("=")


def decode_search_cursor(
    cursor: str,
    resource_type: SearchResourceType,
    context_fingerprint: str,
) -> int:
    if not cursor or len(cursor) > 512:
        raise ValueError("cursor has an invalid length")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        payload: Any = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor is not valid") from exc

    if not isinstance(payload, dict) or payload.get("v") != CURSOR_VERSION:
        raise ValueError("cursor version is not supported")
    if payload.get("r") != resource_type.value:
        raise ValueError("cursor belongs to another resource type")

    cursor_context = payload.get("c")
    if not isinstance(cursor_context, str) or not secrets.compare_digest(
        cursor_context, context_fingerprint
    ):
        raise ValueError("cursor belongs to another search context")

    offset = payload.get("o")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("cursor offset is invalid")
    if offset < 1 or offset > MAX_SEARCH_OFFSET:
        raise ValueError("cursor offset is outside the supported range")
    return offset
