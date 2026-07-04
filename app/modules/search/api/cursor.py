import base64
import hashlib
import json
import secrets
from typing import Any

from app.modules.search.domain.models import SearchResourceType
from app.shared.nlp.preprocessing.text import prepare_for_embedding


CURSOR_VERSION = 1
MAX_SEARCH_OFFSET = 1000


def encode_search_cursor(
    resource_type: SearchResourceType,
    normalized_query: str,
    offset: int,
) -> str:
    payload = {
        "v": CURSOR_VERSION,
        "r": resource_type.value,
        "q": _query_fingerprint(normalized_query),
        "o": offset,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii").rstrip("=")


def decode_search_cursor(
    cursor: str,
    resource_type: SearchResourceType,
    query: str,
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

    expected_query = _query_fingerprint(prepare_for_embedding(query))
    cursor_query = payload.get("q")
    if not isinstance(cursor_query, str) or not secrets.compare_digest(
        cursor_query, expected_query
    ):
        raise ValueError("cursor belongs to another query")

    offset = payload.get("o")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("cursor offset is invalid")
    if offset < 1 or offset > MAX_SEARCH_OFFSET:
        raise ValueError("cursor offset is outside the supported range")
    return offset


def _query_fingerprint(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:24]
