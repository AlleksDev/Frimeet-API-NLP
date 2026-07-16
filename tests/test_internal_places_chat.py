import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.shared.config.settings import get_settings


AUTHORIZATION = {"Authorization": "Bearer test-nlp-service-token"}
BASE_REQUEST = {
    "conversation_id": "cb3456ef-598e-49d9-9bf9-b2ba99055ad7",
    "turn": 1,
    "message": "recomiendame una cafeteria",
    "state": {},
    "user_location": {"lat": 16.7531, "lng": -93.1156},
    "candidate_limit": 5,
    "result_limit": 3,
}


def test_internal_chat_requires_service_bearer() -> None:
    client = TestClient(create_app())

    missing = client.post("/internal/places/chat", json=BASE_REQUEST)
    invalid = client.post(
        "/internal/places/chat",
        json=BASE_REQUEST,
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_internal_chat_returns_only_technical_candidates() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/internal/places/chat",
        json=BASE_REQUEST,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert payload["location_directive"] == {
        "source": "user_current",
        "scope": "user_current_location",
        "anchor_place_id": None,
        "anchor_text": None,
        "radius_meters": None,
        "strict_radius": False,
    }
    assert payload["candidates"]
    assert {candidate["place_id"] for candidate in payload["candidates"]} == {
        "place_1"
    }
    assert set(payload["candidates"][0]) == {
        "place_id",
        "content_score",
        "semantic_score",
        "lexical_score",
        "match_level",
        "matched_reasons",
    }
    assert payload["metadata"]["used_llm"] is False
    assert payload["metadata"]["category_source"] == "explicit"
    assert payload["trace_id"].startswith("trace_")


def test_internal_chat_recommends_restaurants_for_implicit_food_intent() -> None:
    client = TestClient(create_app())
    request = {**BASE_REQUEST, "message": "quiero comer algo"}

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert payload["state_patch"]["target_category"] == "restaurant"
    assert payload["metadata"]["category_source"] == "lexical_activity"
    assert payload["unresolved"] == []
    assert payload["candidates"]


@pytest.mark.parametrize(
    ("message", "expected_place_id"),
    (
        ("parque", "place_6"),
        ("quiero ir al parque", "place_6"),
        ("compras", "place_9"),
    ),
)
def test_internal_chat_recommends_for_short_explicit_categories(
    message: str,
    expected_place_id: str,
) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/internal/places/chat",
        json={**BASE_REQUEST, "message": message},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert expected_place_id in {
        candidate["place_id"] for candidate in payload["candidates"]
    }


def test_internal_chat_never_returns_parks_for_a_cafe_query() -> None:
    client = TestClient(create_app())
    request = {
        **BASE_REQUEST,
        "message": "recomiendame alguna cafeteria cerca del Parque Cana Hueca",
    }

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert {candidate["place_id"] for candidate in payload["candidates"]} == {
        "place_1"
    }
    assert payload["location_directive"]["source"] == "explicit_anchor"
    assert payload["location_directive"]["anchor_place_id"] == "place_6"


def test_internal_chat_asks_about_ambiguous_reference_scope() -> None:
    client = TestClient(create_app())
    request = {
        **BASE_REQUEST,
        "message": "cafeterias como la de Hello Kitty cerca del Parque Central",
    }

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "clarification"
    assert payload["candidates"] == []
    assert payload["unresolved"] == ["location_scope"]
    assert payload["clarification"]["kind"] == "location_scope"
    assert [option["id"] for option in payload["clarification"]["options"]] == [
        "target_results",
        "reference_entity",
    ]


def test_internal_chat_accepts_a_stateful_continuation() -> None:
    client = TestClient(create_app())
    request = {
        **BASE_REQUEST,
        "turn": 2,
        "message": "mas barato y no tan lejos",
        "state": {
            "target_category": "cafe",
            "soft_preferences": ["tranquilo"],
            "taxonomy_version": "places-taxonomy-v1",
        },
    }

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert payload["state_patch"]["hard_filters"] == {
        "price_preference": "lower"
    }
    assert payload["location_directive"]["source"] == "user_current"


def test_internal_chat_requires_current_user_location() -> None:
    client = TestClient(create_app())
    request = {key: value for key, value in BASE_REQUEST.items() if key != "user_location"}

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 422


def test_internal_chat_resolves_a_pending_scope_choice() -> None:
    client = TestClient(create_app())
    first_request = {
        **BASE_REQUEST,
        "message": (
            "cafeterias como la de Hello Kitty cerca del Parque Cana Hueca"
        ),
    }
    first_response = client.post(
        "/internal/places/chat",
        json=first_request,
        headers=AUTHORIZATION,
    )
    first_payload = first_response.json()
    second_request = {
        **BASE_REQUEST,
        "turn": 2,
        "message": "la primera opcion",
        "state": first_payload["state_patch"],
    }

    second_response = client.post(
        "/internal/places/chat",
        json=second_request,
        headers=AUTHORIZATION,
    )

    assert first_response.status_code == 200
    assert first_payload["action"] == "clarification"
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["action"] == "recommendations"
    assert second_payload["state_patch"]["pending_clarification"] is None
    assert second_payload["location_directive"]["source"] == "explicit_anchor"
    assert second_payload["location_directive"]["anchor_place_id"] == "place_6"


def test_internal_chat_resolves_a_structured_scope_choice() -> None:
    client = TestClient(create_app())
    first_response = client.post(
        "/internal/places/chat",
        json={
            **BASE_REQUEST,
            "message": (
                "cafeterias como la de Hello Kitty cerca del Parque Cana Hueca"
            ),
        },
        headers=AUTHORIZATION,
    )
    first_payload = first_response.json()
    clarification = first_payload["clarification"]

    second_response = client.post(
        "/internal/places/chat",
        json={
            **BASE_REQUEST,
            "turn": 2,
            "message": "Buscar cerca del Parque Cana Hueca",
            "clarification_choice": {
                "clarification_id": clarification["id"],
                "option_id": "target_results",
            },
            "state": first_payload["state_patch"],
        },
        headers=AUTHORIZATION,
    )

    assert first_payload["action"] == "clarification"
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["action"] == "recommendations"
    assert second_payload["clarification"] is None
    assert second_payload["state_patch"]["pending_clarification"] is None


def test_internal_chat_rejects_a_stale_structured_choice() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/internal/places/chat",
        json={
            **BASE_REQUEST,
            "message": "Restaurantes",
            "clarification_choice": {
                "clarification_id": "00000000-0000-4000-8000-000000000000",
                "option_id": "restaurant",
            },
        },
        headers=AUTHORIZATION,
    )

    assert response.status_code == 409


def test_internal_chat_applies_content_exclusions_before_ranking() -> None:
    client = TestClient(create_app())
    request = {
        **BASE_REQUEST,
        "message": "una cafeteria sin tranquilo",
    }

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "no_match"
    assert payload["candidates"] == []
    assert payload["state_patch"]["exclusions"] == ["tranquilo"]


def test_internal_chat_rejects_an_incompatible_taxonomy_state() -> None:
    client = TestClient(create_app())
    request = {
        **BASE_REQUEST,
        "state": {"taxonomy_version": "places-taxonomy-old"},
    }

    response = client.post(
        "/internal/places/chat",
        json=request,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 409


def test_internal_chat_feature_flag_is_off_safe(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "places_chat_v2_enabled", False)
    client = TestClient(create_app())

    response = client.post(
        "/internal/places/chat",
        json=BASE_REQUEST,
        headers=AUTHORIZATION,
    )

    assert response.status_code == 404
