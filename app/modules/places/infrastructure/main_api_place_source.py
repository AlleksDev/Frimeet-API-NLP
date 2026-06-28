from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from app.shared.config.settings import Settings
from app.shared.content_hash import stable_content_hash
from app.shared.nlp.preprocessing.text import clean_text
from app.modules.places.infrastructure.place_semantic_document import (
    PLACE_SEMANTIC_DOCUMENT_VERSION,
    build_place_semantic_document,
    resolve_place_tags,
)


@dataclass(frozen=True)
class PlaceSourceRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    content_hash: str
    is_active: bool


class MainApiPlacesClient:
    """Reads real places from the main product API for offline embedding jobs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_places_search_path.lstrip("/")

    async def iter_places(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PlaceSourceRecord]:
        limit = page_limit or self._settings.main_api_places_page_limit
        page = 1
        offset = 0
        cursor: str | None = None
        headers = self._build_headers()
        seen_ids: set[str] = set()
        pagination_mode = self._settings.main_api_places_pagination_mode

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            while True:
                params = self._build_pagination_params(
                    limit=limit,
                    page=page,
                    offset=offset,
                    cursor=cursor,
                )
                response = await client.get(self._path, params=params)
                response.raise_for_status()
                payload = response.json()
                places = self._extract_places(payload)

                if not places:
                    break

                yielded_this_page = 0
                for place in places:
                    record = place_to_source_record(place)
                    if record is not None and record.id not in seen_ids:
                        seen_ids.add(record.id)
                        yielded_this_page += 1
                        yield record

                if yielded_this_page == 0 or (max_pages is not None and page >= max_pages):
                    break

                if pagination_mode == "cursor":
                    cursor = self._extract_next_cursor(payload)
                    if not self._extract_has_more(payload) or not cursor:
                        break
                elif len(places) < limit:
                    break

                page += 1
                offset += limit

    def _build_headers(self) -> dict[str, str]:
        token = self._settings.main_api_internal_token or self._settings.main_api_auth_token
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _build_pagination_params(
        self,
        limit: int,
        page: int,
        offset: int,
        cursor: str | None,
    ) -> dict[str, int | str]:
        params: dict[str, int | str] = {"limit": limit}
        mode = self._settings.main_api_places_pagination_mode
        if mode == "cursor":
            if cursor:
                params["cursor"] = cursor
        elif mode == "offset":
            params["offset"] = offset
        else:
            params["page"] = page
        return params

    @staticmethod
    def _extract_places(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        candidates = [
            payload.get("data"),
            payload.get("places"),
            payload.get("items"),
            payload.get("results"),
        ]
        for candidate in candidates:
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
            if isinstance(candidate, dict):
                nested = MainApiPlacesClient._extract_places(candidate)
                if nested:
                    return nested

        return []

    @staticmethod
    def _extract_next_cursor(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        cursor = payload.get("next_cursor") or payload.get("nextCursor")
        if cursor is None:
            return None
        cursor = str(cursor).strip()
        return cursor or None

    @staticmethod
    def _extract_has_more(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("has_more", payload.get("hasMore", False))
        return bool(value)


def place_to_source_record(place: dict[str, Any]) -> PlaceSourceRecord | None:
    place_id = _first_present(place, "id", "_id", "place_id", "uuid")
    if place_id is None:
        return None

    name = str(_first_present(place, "name", "title", default="")).strip()
    category = _first_present(place, "category", "type")
    city = _first_present(place, "city", "municipality")
    state = _first_present(place, "state", default="Chiapas")
    source = _first_present(place, "source")
    price_range = _first_present(place, "price_range", "priceRange")
    is_active = _first_present(place, "is_active", "isActive", default=True)
    description = str(_first_present(place, "description", "summary", "about", default=""))
    address = str(_first_present(place, "address", "formatted_address", default=""))

    resolved_tags = resolve_place_tags(
        _first_present(place, "tags", "keywords", default=[])
    )
    occasion = _as_text_list(_first_present(place, "occasion", "occasions", default=[]))

    document = build_place_semantic_document(
        name=name,
        category=category,
        description=description,
        resolved_tags=resolved_tags,
    )

    metadata = {
        "name": name,
        "category": _to_metadata_value(category),
        "city": _to_metadata_value(city),
        "state": _to_metadata_value(state),
        "source": _to_metadata_value(source),
        "price_range": _to_metadata_value(price_range),
        "is_active": bool(is_active),
        "occasion": ",".join(occasion),
        "tags": ",".join(resolved_tags.names),
        "tag_ids": list(resolved_tags.ids),
        "tag_categories": list(resolved_tags.categories),
        "unknown_tag_ids": list(resolved_tags.unknown_ids),
        "semantic_document_version": PLACE_SEMANTIC_DOCUMENT_VERSION,
        "short_description": description[:300],
    }
    filtered_metadata = {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [], ())
    }
    content_hash = stable_content_hash(
        {
            "document": document,
            "metadata": filtered_metadata,
            "is_active": bool(is_active),
            "semantic_document_version": PLACE_SEMANTIC_DOCUMENT_VERSION,
        }
    )

    return PlaceSourceRecord(
        id=str(place_id),
        document=document,
        metadata=filtered_metadata,
        content_hash=content_hash,
        is_active=bool(is_active),
    )


def _first_present(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _to_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
