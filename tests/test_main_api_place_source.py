from types import SimpleNamespace

import httpx
import pytest

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


def test_place_to_source_record_maps_category_label_and_closed_state() -> None:
    record = place_to_source_record(
        {
            "id": "place_closed",
            "name": "Lugar cerrado",
            "category": "religious_organization",
            "categoryLabel": "Organización religiosa",
            "isActive": True,
            "isPermanentlyClosed": True,
        }
    )

    assert record is not None
    assert record.metadata["category_label"] == "Organización religiosa"
    assert record.metadata["is_active"] is False
    assert record.is_active is False


def test_explicit_string_active_state_is_not_coerced_as_truthy() -> None:
    record = place_to_source_record(
        {
            "id": "place_inactive",
            "name": "Lugar inactivo",
            "is_active": "false",
        }
    )

    assert record is not None
    assert record.is_active is False


def test_place_to_source_record_resolves_tags_without_token_repetition() -> None:
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
    assert record.metadata["semantic_document_version"] == "structured-place-v4"
    assert record.document.count("Nombre Ambiguo") == 1
    assert record.document.count("Venta de prendas y accesorios") == 1
    assert record.document.count("Compras Ropa barata") == 1
    assert "Categoria: shopping." in record.document
    assert "osm" not in record.document
    assert "Direccion que no debe influir" not in record.document
    assert "9999" not in record.document


def test_semantic_documents_keep_source_category_without_manual_expansion() -> None:
    park = place_to_source_record(
        {"id": "park", "name": "Area Uno", "category": "park"}
    )
    shopping = place_to_source_record(
        {"id": "shopping", "name": "Area Dos", "category": "shopping"}
    )

    assert park is not None and "Categoria: park." in park.document
    assert shopping is not None and "Categoria: shopping." in shopping.document
    assert "park parque naturaleza" not in park.document
    assert "shopping compras tiendas" not in shopping.document


def test_place_source_uses_spanish_category_and_structured_facets() -> None:
    record = place_to_source_record(
        {
            "id": "aquatic_center",
            "name": "Centro Acuatico Municipal",
            "description": None,
            "category": "sports_center",
            "attributes": {
                "has_restrooms": True,
                "has_parking": False,
                "has_screens": True,
                "has_entertainment": None,
                "kid_friendly": None,
                "good_for_cooling_off": "yes",
            },
            "entertainment": [
                {"feature_key": "live_music", "label": "Musica en vivo"}
            ],
            "contained_items": [
                {
                    "name": "Alberca semiolimpica",
                    "category": "actividad acuática",
                    "description": "Carriles para natacion",
                }
            ],
            "menu": {
                "items": [
                    {
                        "name": "Agua fresca",
                        "category": "bebidas",
                        "description": "Bebida fria",
                    }
                ]
            },
        },
        category_labels={"sports_center": "Centro deportivo"},
    )

    assert record is not None
    assert record.metadata["category"] == "sports_center"
    assert record.metadata["category_label"] == "Centro deportivo"
    assert record.metadata["semantic_document_version"] == "structured-place-v4"
    assert record.metadata["attribute_states"] == {
        "has_restrooms": True,
        "has_parking": False,
        "good_for_cooling_off": "yes",
        "has_screens": True,
    }
    assert "banos y sanitarios" in record.metadata["attribute_terms"]
    assert "pantallas" in record.metadata["attribute_terms"]
    assert any(
        "refrescarse" in term and "nadar" in term
        for term in record.metadata["attribute_terms"]
    )
    assert record.metadata["negative_attribute_terms"] == ["estacionamiento"]
    assert "has_entertainment" not in record.metadata["attribute_states"]
    assert "kid_friendly" not in record.metadata["attribute_states"]
    assert record.metadata["entertainment_features"] == [
        "Musica en vivo live music"
    ]
    assert record.metadata["contained_items"] == [
        "Alberca semiolimpica actividad acuática Carriles para natacion"
    ]
    assert record.metadata["menu_items"] == [
        "Agua fresca bebidas Bebida fria"
    ]
    assert record.metadata["source_fields_present"] == [
        "name",
        "category",
        "facets",
    ]

    assert "Categoria: Centro deportivo sports center." in record.document
    assert "Atributos confirmados:" in record.document
    assert "banos y sanitarios" in record.document
    assert "pantallas" in record.document
    assert "refrescarse" in record.document
    assert "Musica en vivo live music" in record.document
    assert "Alberca semiolimpica" in record.document
    assert "Agua fresca" in record.document
    assert "estacionamiento" not in record.document
    assert "Descripcion:" not in record.document


def test_extract_category_labels_accepts_main_api_contract() -> None:
    labels = MainApiPlacesClient._extract_category_labels(
        {
            "data": [
                {"value": "sports", "label": "Deportes"},
                {"value": "sports_center", "label": "Centro deportivo"},
                {"value": "", "label": "Invalida"},
                {"value": "park", "label": ""},
                "not-an-object",
            ]
        }
    )

    assert labels == {
        "sports": "Deportes",
        "sports_center": "Centro deportivo",
    }


def _place_source_client() -> MainApiPlacesClient:
    return MainApiPlacesClient(
        SimpleNamespace(
            main_api_base_url="https://main-api.test",
            main_api_places_snapshot_path="/api/v1/internal/places/snapshot",
            main_api_places_changes_path="/api/v1/internal/places/changes",
            main_api_place_categories_path="/api/v1/places/categories",
            main_api_place_catalog_language="es",
            main_api_internal_token="internal-token",
        )
    )


@pytest.mark.asyncio
async def test_category_catalog_request_uses_spanish_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/places/categories"
        assert request.url.params["lang"] == "es"
        return httpx.Response(
            200,
            json={"data": [{"value": "park", "label": "Parque"}]},
        )

    async with httpx.AsyncClient(
        base_url="https://main-api.test",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        labels = await _place_source_client()._load_category_labels(http_client)

    assert labels == {"park": "Parque"}


@pytest.mark.asyncio
async def test_category_catalog_failure_aborts_sync_instead_of_reindexing_english() -> None:
    async with httpx.AsyncClient(
        base_url="https://main-api.test",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"error": "unavailable"})
        ),
    ) as http_client:
        with pytest.raises(RuntimeError, match="Spanish place category catalog"):
            await _place_source_client()._load_category_labels(http_client)


def test_place_source_does_not_index_unavailable_menu_items() -> None:
    record = place_to_source_record(
        {
            "id": "restaurant_menu",
            "name": "Restaurante",
            "category": "restaurant",
            "menu": [
                {"name": "Sopa disponible", "is_available": True},
                {"name": "Platillo agotado", "is_available": False},
            ],
        }
    )

    assert record is not None
    assert record.metadata["menu_items"] == ["Sopa disponible"]
    assert "Sopa disponible" in record.document
    assert "Platillo agotado" not in record.document


def test_explicit_no_entertainment_overrides_stale_feature_rows() -> None:
    record = place_to_source_record(
        {
            "id": "quiet_place",
            "name": "Lugar tranquilo",
            "category": "park",
            "attributes": {
                "has_entertainment": False,
                "entertainment_note": "Musica en vivo los viernes",
            },
            "entertainment": [
                {"feature_key": "live_music", "label": "Musica en vivo"}
            ],
        }
    )

    assert record is not None
    assert record.metadata["attribute_states"]["has_entertainment"] is False
    assert record.metadata["negative_attribute_terms"] == ["entretenimiento"]
    assert "entertainment_features" not in record.metadata
    assert "Musica en vivo" not in record.document


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


def test_place_source_extracts_nested_cursor_metadata() -> None:
    payload = {
        "data": {
            "items": [{"id": "place_1"}],
            "pagination": {
                "nextCursor": "cursor_nested",
                "hasMore": True,
            },
        }
    }

    assert MainApiPlacesClient._extract_next_cursor(payload) == "cursor_nested"
    assert MainApiPlacesClient._extract_has_more(payload) is True


def test_place_source_uses_internal_snapshot_and_service_token() -> None:
    client = _place_source_client()

    assert client._path == "api/v1/internal/places/snapshot"
    assert client._changes_path == "api/v1/internal/places/changes"
    assert client._build_headers() == {
        "Authorization": "Bearer internal-token"
    }
