from fastapi.testclient import TestClient

from app.main import create_app


def test_places_search_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/search",
        json={
            "query": "lugares tranquilos para cenar",
            "city": "Tuxtla Gutierrez",
            "filters": {"is_active": True},
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "lugares tranquilos para cenar"
    assert payload["places"]


def test_places_chat_endpoint_returns_trace_and_structured_places() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/chat",
        json={
            "message": "quiero una cena tranquila con mi pareja",
            "city": "Tuxtla Gutierrez",
            "filters": {"occasion": "pareja", "is_active": True},
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_id"].startswith("resp_")
    assert payload["nlp_trace_id"].startswith("trace_")
    assert payload["message"]
    assert payload["places"]
    assert payload["metadata"]["places_used_as_context"]


def test_posts_recommendations_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/posts/recommendations",
        json={
            "query": "ideas para fotos con amigos",
            "city": "Tuxtla Gutierrez",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["posts"]
    assert payload["metadata"]["computed_clusters_during_request"] is False


def test_posts_clusters_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/posts/clusters")

    assert response.status_code == 200
    payload = response.json()
    assert payload["clusters"]
    assert payload["metadata"]["computed_during_request"] is False
