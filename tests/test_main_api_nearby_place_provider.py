from urllib.parse import parse_qs

import httpx
import pytest

from app.modules.places.infrastructure.main_api_nearby_place_provider import (
    MainApiNearbyPlaceProvider,
)
from app.shared.config.settings import Settings


@pytest.mark.asyncio
async def test_nearby_provider_calls_main_api_and_extracts_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        assert request.url.path == "/api/v1/places/nearby"
        assert query["lat"] == ["16.7531"]
        assert query["lng"] == ["-93.1156"]
        assert query["radius"] == ["10000"]
        return httpx.Response(
            200,
            json=[
                {"id": "place_2", "name": "Mirador"},
                {"uuid": "place_6", "name": "Parque"},
            ],
        )

    settings = Settings(
        _env_file=None,
        MAIN_API_BASE_URL="https://main-api.test",
        MAIN_API_PLACES_NEARBY_PATH="/api/v1/places/nearby",
    )
    provider = MainApiNearbyPlaceProvider(
        settings=settings,
        transport=httpx.MockTransport(handler),
    )

    ids = await provider.get_nearby_place_ids(
        latitude=16.7531,
        longitude=-93.1156,
        radius_meters=10_000,
    )

    assert ids == {"place_2", "place_6"}
