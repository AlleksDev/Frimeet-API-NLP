import json

import httpx
import pytest

from app.modules.places.infrastructure.main_api_place_anchor_resolver import (
    MainApiPlaceAnchorResolver,
)
from app.shared.config.settings import Settings
from app.shared.errors.exceptions import AppError


@pytest.mark.asyncio
async def test_anchor_resolver_calls_the_internal_main_api_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/internal/places/resolve-anchor"
        assert request.headers["Authorization"] == "Bearer main-api-token"
        assert json.loads(request.content) == {
            "query": "Parque Central",
            "city": "Tuxtla Gutierrez",
            "state": "Chiapas",
            "limit": 3,
        }
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": "place_park",
                            "name": "Parque Central",
                            "category": "park",
                            "lat": 0,
                            "lng": 0,
                            "tags": "familia,aire libre",
                            "score": 0.95,
                        }
                    ]
                }
            },
        )

    settings = Settings(
        _env_file=None,
        MAIN_API_BASE_URL="https://main-api.test",
        MAIN_API_INTERNAL_TOKEN="main-api-token",
        MAIN_API_PLACE_ANCHOR_RESOLVE_PATH=(
            "/api/v1/internal/places/resolve-anchor"
        ),
    )
    resolver = MainApiPlaceAnchorResolver(
        settings=settings,
        transport=httpx.MockTransport(handler),
    )

    anchors = await resolver.resolve(
        text="Parque Central",
        city="Tuxtla Gutierrez",
        state="Chiapas",
        limit=3,
    )

    assert len(anchors) == 1
    assert anchors[0].place_id == "place_park"
    assert anchors[0].latitude == 0.0
    assert anchors[0].longitude == 0.0
    assert anchors[0].attributes == ("familia", "aire libre")


@pytest.mark.asyncio
async def test_anchor_resolver_maps_main_api_failures_to_app_error() -> None:
    settings = Settings(
        _env_file=None,
        MAIN_API_BASE_URL="https://main-api.test",
        MAIN_API_INTERNAL_TOKEN="main-api-token",
    )
    resolver = MainApiPlaceAnchorResolver(
        settings=settings,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(503, json={"detail": "unavailable"})
        ),
    )

    with pytest.raises(AppError) as error:
        await resolver.resolve(
            text="Parque Central",
            city=None,
            state=None,
        )

    assert error.value.code == "place_anchor_resolver_unavailable"
    assert error.value.status_code == 502
