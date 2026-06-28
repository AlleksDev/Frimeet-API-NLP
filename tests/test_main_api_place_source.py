from app.modules.places.infrastructure.main_api_place_source import (
    MainApiPlacesClient,
    place_to_source_record,
)
from app.modules.places.infrastructure.place_semantic_document import (
    place_tag_catalog,
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


def test_place_to_source_record_resolves_and_weights_numeric_tags() -> None:
    record = place_to_source_record(
        {
            "id": "place_weighted",
            "name": "Nombre Ambiguo",
            "category": "shopping",
            "source": "osm",
            "address": "Direccion que no debe influir",
            "tags": [29, 187, 9999],
            "description": "Venta de prendas y accesorios.",
        }
    )

    assert record is not None
    assert record.metadata["tags"] == "Compras,Ropa barata"
    assert record.metadata["tag_ids"] == [29, 187, 9999]
    assert record.metadata["unknown_tag_ids"] == [9999]
    assert record.metadata["semantic_document_version"] == "weighted-tags-v2"
    assert record.document.count("Nombre Ambiguo") == 1
    assert record.document.count("Venta de prendas y accesorios.") == 3
    assert record.document.count("Compras Ropa barata") == 6
    assert record.document.count("compras tiendas ropa") == 4
    assert "osm" not in record.document
    assert "Direccion que no debe influir" not in record.document
    assert "9999" not in record.document


def test_place_tag_catalog_contains_complete_supplied_mapping() -> None:
    catalog = place_tag_catalog()

    assert len(catalog) == 310
    assert catalog[12].name == "Tranquilo"
    assert catalog[33].name == "Café"
    assert catalog[187].name == "Ropa barata"
    assert catalog[250].name == "museum"


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
