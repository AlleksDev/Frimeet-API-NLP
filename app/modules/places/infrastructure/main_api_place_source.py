from dataclasses import dataclass
import logging
from collections.abc import Mapping
from typing import Any, AsyncIterator

import httpx

from app.shared.config.settings import Settings
from app.shared.content_hash import stable_content_hash
from app.shared.nlp.preprocessing.text import clean_text
from app.modules.places.infrastructure.place_semantic_document import (
    PLACE_SEMANTIC_DOCUMENT_VERSION,
    ResolvedPlaceTags,
    build_place_semantic_document,
    resolve_place_tags,
)
from app.modules.places.infrastructure.place_facets import resolve_place_facets


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaceSourceRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    content_hash: str
    is_active: bool


@dataclass(frozen=True)
class PlaceChangeRecord:
    event_id: int
    place_id: str
    operation: str
    place: PlaceSourceRecord | None


class MainApiPlacesClient:
    """Reads real places from the main product API for offline embedding jobs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_places_snapshot_path.lstrip("/")
        self._changes_path = settings.main_api_places_changes_path.lstrip("/")
        self._categories_path = settings.main_api_place_categories_path.lstrip("/")
        self._catalog_language = settings.main_api_place_catalog_language

    async def iter_places(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PlaceSourceRecord]:
        limit = page_limit or self._settings.main_api_places_page_limit
        page = 1
        cursor: str | None = None
        headers = self._build_headers()
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            category_labels = await self._load_category_labels(client)
            while True:
                params = self._build_pagination_params(
                    limit=limit,
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
                    record = place_to_source_record(
                        place,
                        category_labels=category_labels,
                    )
                    if record is not None and record.id not in seen_ids:
                        seen_ids.add(record.id)
                        yielded_this_page += 1
                        yield record

                if yielded_this_page == 0 or (max_pages is not None and page >= max_pages):
                    break

                cursor = self._extract_next_cursor(payload)
                if not self._extract_has_more(payload) or not cursor:
                    break

                page += 1

    async def iter_changes(
        self,
        after_event_id: int,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PlaceChangeRecord]:
        limit = page_limit or self._settings.main_api_places_page_limit
        current_event_id = max(after_event_id, 0)
        page = 1
        headers = self._build_headers()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            category_labels = await self._load_category_labels(client)
            while True:
                response = await client.get(
                    self._changes_path,
                    params={"after_id": current_event_id, "limit": limit},
                )
                response.raise_for_status()
                payload = response.json()
                changes = self._extract_places(payload)
                if not changes:
                    break

                for change in changes:
                    event_id = _as_int(change.get("event_id"))
                    place_id = str(change.get("place_id") or "").strip()
                    operation = str(change.get("operation") or "upsert").strip().lower()
                    raw_place = change.get("place")
                    place = (
                        place_to_source_record(
                            raw_place,
                            category_labels=category_labels,
                        )
                        if isinstance(raw_place, dict)
                        else None
                    )
                    if event_id is None or not place_id:
                        raise ValueError("invalid place change returned by main API")
                    current_event_id = max(current_event_id, event_id)
                    yield PlaceChangeRecord(
                        event_id=event_id,
                        place_id=place_id,
                        operation=operation,
                        place=place,
                    )

                if not self._extract_has_more(payload):
                    break
                if max_pages is not None and page >= max_pages:
                    break
                page += 1

    async def _load_category_labels(
        self,
        client: httpx.AsyncClient,
    ) -> dict[str, str]:
        """Load localized labels once per sync, avoiding one request per place."""

        try:
            response = await client.get(
                self._categories_path,
                params={"lang": self._catalog_language},
            )
            response.raise_for_status()
            labels = self._extract_category_labels(response.json())
            if not labels:
                raise ValueError("the localized category catalog is empty")
            return labels
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Do not replace an already-localized index with English fallbacks
            # because of a transient catalog failure. Deploy the main API first
            # and retry this derived-data sync when its Spanish catalog is ready.
            logger.error(
                "Could not load the Spanish place category catalog: %s",
                exc,
            )
            raise RuntimeError(
                "the Spanish place category catalog is required for place sync"
            ) from exc

    @staticmethod
    def _extract_category_labels(payload: Any) -> dict[str, str]:
        if not isinstance(payload, Mapping):
            return {}
        records: Any = payload.get("data", payload.get("categories", []))
        if isinstance(records, Mapping):
            records = records.get("items", records.get("data", []))
        if not isinstance(records, list):
            return {}

        labels: dict[str, str] = {}
        for item in records:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get("value") or "").strip()
            label = str(item.get("label") or "").strip()
            if value and label:
                labels[value.casefold()] = label
        return labels

    def _build_headers(self) -> dict[str, str]:
        token = self._settings.main_api_internal_token
        if not token:
            raise RuntimeError(
                "MAIN_API_INTERNAL_TOKEN is required for the internal Places snapshot"
            )
        return {"Authorization": f"Bearer {token}"}

    def _build_pagination_params(
        self,
        limit: int,
        cursor: str | None,
    ) -> dict[str, int | str]:
        params: dict[str, int | str] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
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
        if cursor is not None:
            cursor = str(cursor).strip()
            if cursor:
                return cursor
        for key in ("data", "pagination", "meta"):
            nested_cursor = MainApiPlacesClient._extract_next_cursor(
                payload.get(key)
            )
            if nested_cursor:
                return nested_cursor
        return None

    @staticmethod
    def _extract_has_more(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "has_more" in payload or "hasMore" in payload:
            value = payload.get("has_more", payload.get("hasMore"))
            return bool(value)
        for key in ("data", "pagination", "meta"):
            nested = payload.get(key)
            if (
                isinstance(nested, dict)
                and MainApiPlacesClient._extract_has_more(nested)
            ):
                return True
        return False


def place_to_source_record(
    place: dict[str, Any],
    *,
    category_labels: Mapping[str, str] | None = None,
) -> PlaceSourceRecord | None:
    place_id = _first_present(place, "id", "_id", "place_id", "uuid")
    if place_id is None:
        return None

    name = str(_first_present(place, "name", "title", default="")).strip()
    category = _first_present(place, "category", "type")
    source_category_label = _first_present(
        place,
        "category_label",
        "categoryLabel",
    )
    category_key = str(category or "").strip().casefold()
    category_label = (
        (category_labels or {}).get(category_key)
        or source_category_label
    )
    city = _first_present(place, "city", "municipality")
    state = _first_present(place, "state", default="Chiapas")
    source = _first_present(place, "source")
    price_range = _first_present(place, "price_range", "priceRange")
    explicit_is_active = _first_present(place, "is_active", "isActive")
    is_permanently_closed = _as_bool(
        _first_present(
            place,
            "is_permanently_closed",
            "isPermanentlyClosed",
            default=False,
        ),
        default=False,
    )
    if is_permanently_closed:
        is_active = False
    elif explicit_is_active is None:
        is_active = True
    else:
        is_active = _as_bool(explicit_is_active, default=True)
    description = str(_first_present(place, "description", "summary", "about", default=""))
    address = str(_first_present(place, "address", "formatted_address", default=""))

    raw_tags = _first_present(place, "tag_ids", "tags", "keywords", default=[])
    resolved_source_tags = resolve_place_tags(raw_tags)
    localized_tag_values = _first_present(place, "tag_labels", "tagLabels")
    resolved_localized_tags = (
        resolve_place_tags(localized_tag_values)
        if localized_tag_values is not None
        else resolved_source_tags
    )
    resolved_tags = ResolvedPlaceTags(
        names=resolved_localized_tags.names,
        ids=resolved_source_tags.ids,
        categories=resolved_source_tags.categories,
        unknown_ids=resolved_source_tags.unknown_ids,
    )
    resolved_facets = resolve_place_facets(
        _first_present(place, "attributes", default={}),
        _first_present(
            place,
            "entertainment",
            "entertainment_features",
            "entertainmentFeatures",
            default=[],
        ),
        _first_present(
            place,
            "contained_items",
            "containedItems",
            default=[],
        ),
        _first_present(place, "menu", "menu_items", "menuItems", default=[]),
    )
    occasion = _as_text_list(_first_present(place, "occasion", "occasions", default=[]))

    document = build_place_semantic_document(
        name=name,
        category=category,
        description=description,
        resolved_tags=resolved_tags,
        category_label=category_label,
        resolved_facets=resolved_facets,
    )

    metadata = {
        "name": name,
        "category": _to_metadata_value(category),
        "category_label": _to_metadata_value(category_label),
        "city": _to_metadata_value(city),
        "state": _to_metadata_value(state),
        "source": _to_metadata_value(source),
        "price_range": _to_metadata_value(price_range),
        "is_active": is_active,
        "occasion": ",".join(occasion),
        "tags": ",".join(resolved_tags.names),
        "tag_ids": list(resolved_tags.ids),
        "tag_categories": list(resolved_tags.categories),
        "unknown_tag_ids": list(resolved_tags.unknown_ids),
        "attribute_states": resolved_facets.attribute_states,
        "attribute_terms": list(resolved_facets.positive_attributes),
        "negative_attribute_terms": list(resolved_facets.negative_attributes),
        "entertainment_features": list(resolved_facets.entertainment_features),
        "contained_items": list(resolved_facets.contained_items),
        "menu_items": list(resolved_facets.menu_items),
        "source_fields_present": _source_fields_present(
            name=name,
            category=category,
            description=description,
            tags=resolved_tags.names,
            facets=resolved_facets.positive_evidence,
        ),
        "semantic_document_version": PLACE_SEMANTIC_DOCUMENT_VERSION,
        "short_description": description[:300],
    }
    filtered_metadata = {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [], (), {})
    }
    content_hash = stable_content_hash(
        {
            "document": document,
            "metadata": filtered_metadata,
            "is_active": is_active,
            "semantic_document_version": PLACE_SEMANTIC_DOCUMENT_VERSION,
        }
    )

    return PlaceSourceRecord(
        id=str(place_id),
        document=document,
        metadata=filtered_metadata,
        content_hash=content_hash,
        is_active=is_active,
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


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "si", "sí"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _source_fields_present(
    *,
    name: str,
    category: Any,
    description: str,
    tags: tuple[str, ...],
    facets: tuple[str, ...],
) -> list[str]:
    return [
        key
        for key, present in (
            ("name", bool(name.strip())),
            ("category", bool(str(category or "").strip())),
            ("description", bool(description.strip())),
            ("tags", bool(tags)),
            ("facets", bool(facets)),
        )
        if present
    ]
