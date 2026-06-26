from app.modules.places.infrastructure.main_api_place_source import (
    MainApiPlacesClient,
    place_to_source_record,
)


def test_place_to_source_record_maps_api_place() -> None:
    record = place_to_source_record(
        {
            "id": "place_123",
            "name": "Cafe Centro",
            "category": "cafe",
            "city": "Tuxtla Gutierrez",
            "state": "Chiapas",
            "source": "osm",
            "tags": ["cafe", "tranquilo"],
            "description": "Un lugar para platicar.",
        }
    )

    assert record is not None
    assert record.id == "place_123"
    assert "Cafe Centro" in record.document
    assert record.metadata["name"] == "Cafe Centro"
    assert record.metadata["category"] == "cafe"
    assert record.metadata["source"] == "osm"
    assert record.is_active is True
    assert len(record.content_hash) == 64


def test_extract_places_supports_common_response_shapes() -> None:
    payload = {
        "data": {
            "items": [
                {"id": "place_1", "name": "Cafe"},
                {"id": "place_2", "name": "Parque"},
            ]
        }
    }

    places = MainApiPlacesClient._extract_places(payload)

    assert [place["id"] for place in places] == ["place_1", "place_2"]


def test_place_source_extracts_cursor_metadata() -> None:
    payload = {
        "data": [],
        "next_cursor": "cursor_abc",
        "has_more": True,
    }

    assert MainApiPlacesClient._extract_next_cursor(payload) == "cursor_abc"
    assert MainApiPlacesClient._extract_has_more(payload) is True
