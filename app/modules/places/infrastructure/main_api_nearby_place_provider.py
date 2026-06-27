from typing import Any

import httpx

from app.modules.places.application.ports.nearby_place_provider import (
    NearbyPlaceProvider,
)
from app.shared.config.settings import Settings
from app.shared.errors.exceptions import AppError


class MainApiNearbyPlaceProvider(NearbyPlaceProvider):
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_places_nearby_path.lstrip("/")
        self._timeout = settings.main_api_timeout_seconds
        self._transport = transport
        token = settings.main_api_internal_token or settings.main_api_auth_token
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def get_nearby_place_ids(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> set[str]:
        params = {
            "lat": latitude,
            "lng": longitude,
            "radius": radius_meters,
        }
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._headers,
                transport=self._transport,
            ) as client:
                response = await client.get(self._path, params=params)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AppError(
                message="The main API nearby places service is unavailable",
                code="nearby_places_unavailable",
                status_code=502,
            ) from exc

        return self._extract_place_ids(response.json())

    @staticmethod
    def _extract_place_ids(payload: Any) -> set[str]:
        places = MainApiNearbyPlaceProvider._extract_places(payload)
        ids: set[str] = set()
        for place in places:
            place_id = (
                place.get("id")
                or place.get("_id")
                or place.get("place_id")
                or place.get("uuid")
            )
            if place_id is not None:
                ids.add(str(place_id))
        return ids

    @staticmethod
    def _extract_places(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "places", "items", "results"):
            candidate = payload.get(key)
            places = MainApiNearbyPlaceProvider._extract_places(candidate)
            if places:
                return places
        return []
