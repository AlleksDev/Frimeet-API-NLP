from typing import Any, Sequence

import httpx

from app.modules.places.domain.chat_intent import ResolvedPlaceAnchor
from app.modules.places.infrastructure.mock_place_repository import SAMPLE_PLACES
from app.shared.config.settings import Settings
from app.shared.errors.exceptions import AppError
from app.shared.nlp.preprocessing.text import prepare_for_embedding


class MainApiPlaceAnchorResolver:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_place_anchor_resolve_path.lstrip("/")
        self._timeout = settings.main_api_timeout_seconds
        self._transport = transport
        self._headers = (
            {"Authorization": f"Bearer {settings.main_api_internal_token}"}
            if settings.main_api_internal_token
            else {}
        )

    async def resolve(
        self,
        text: str,
        city: str | None,
        state: str | None,
        limit: int = 3,
    ) -> Sequence[ResolvedPlaceAnchor]:
        payload = {
            "query": text,
            "city": city,
            "state": state,
            "limit": limit,
        }
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._headers,
                transport=self._transport,
            ) as client:
                response = await client.post(self._path, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AppError(
                message="The main API place anchor resolver is unavailable",
                code="place_anchor_resolver_unavailable",
                status_code=502,
            ) from exc
        return _extract_anchors(response.json())[:limit]


class MockPlaceAnchorResolver:
    async def resolve(
        self,
        text: str,
        city: str | None,
        state: str | None,
        limit: int = 3,
    ) -> Sequence[ResolvedPlaceAnchor]:
        query = prepare_for_embedding(text)
        matches: list[ResolvedPlaceAnchor] = []
        for place in SAMPLE_PLACES:
            if city and prepare_for_embedding(place["city"]) != prepare_for_embedding(city):
                continue
            if state and prepare_for_embedding(place["state"]) != prepare_for_embedding(state):
                continue
            name = prepare_for_embedding(place["name"])
            if query == name:
                score = 1.0
            elif query in name or name in query:
                score = 0.9
            else:
                query_tokens = set(query.split())
                name_tokens = set(name.split())
                overlap = len(query_tokens & name_tokens) / max(len(query_tokens), 1)
                if overlap < 0.75:
                    continue
                score = overlap
            matches.append(
                ResolvedPlaceAnchor(
                    place_id=str(place["id"]),
                    name=str(place["name"]),
                    category=str(place["category"]),
                    attributes=tuple(str(place.get("tags", "")).split()),
                    score=score,
                )
            )
        return sorted(matches, key=lambda item: (-item.score, item.place_id))[:limit]


def _extract_anchors(payload: Any) -> list[ResolvedPlaceAnchor]:
    items = _extract_items(payload)
    anchors: list[ResolvedPlaceAnchor] = []
    for item in items:
        place_id = item.get("place_id") or item.get("id") or item.get("uuid")
        name = item.get("name") or item.get("title")
        if place_id is None or not name:
            continue
        attributes = item.get("attributes") or item.get("tags") or []
        if isinstance(attributes, str):
            attributes = [part.strip() for part in attributes.split(",") if part.strip()]
        anchors.append(
            ResolvedPlaceAnchor(
                place_id=str(place_id),
                name=str(name),
                category=_optional_string(item.get("category")),
                latitude=_optional_float(_first_present(item, "latitude", "lat")),
                longitude=_optional_float(_first_present(item, "longitude", "lng")),
                attributes=tuple(str(value) for value in attributes),
                score=float(item.get("score") or 0.0),
            )
        )
    return sorted(anchors, key=lambda item: (-item.score, item.place_id))


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "places", "items", "results"):
        nested = _extract_items(payload.get(key))
        if nested:
            return nested
    return []


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None
