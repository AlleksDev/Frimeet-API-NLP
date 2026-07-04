from fastapi.testclient import TestClient

from app.main import create_app


def test_global_search_returns_requested_sections_and_diverse_top_results() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/search",
        json={
            "query": "club universitario ajedrez",
            "resource_types": ["clubs", "users", "events"],
            "per_type_limit": 3,
            "top_limit": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["sections"]) == {"clubs", "users", "events"}
    assert payload["sections"]["clubs"][0]["title"] == "Club de Ajedrez Universitario"
    assert payload["metadata"]["embedding_computed_once"] is True
    assert payload["metadata"]["failed_resources"] == {}
    assert len(payload["top_results"]) <= 5


def test_private_group_requires_authorized_requester() -> None:
    client = TestClient(create_app())
    anonymous = client.post(
        "/search",
        json={"query": "amigos universidad", "resource_types": ["groups"]},
    )
    authorized = client.post(
        "/search",
        headers={"Authorization": "Bearer test-search-token"},
        json={
            "query": "amigos universidad",
            "resource_types": ["groups"],
            "requester_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    assert anonymous.status_code == 200
    assert anonymous.json()["sections"]["groups"] == []
    assert authorized.status_code == 200
    assert authorized.json()["sections"]["groups"][0]["id"] == "group_uni"
    assert "authorized_user_ids" not in authorized.json()["sections"]["groups"][0]["metadata"]


def test_requester_scoped_search_rejects_untrusted_callers() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/search",
        json={
            "query": "amigos universidad",
            "resource_types": ["groups"],
            "requester_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    assert response.status_code == 403
