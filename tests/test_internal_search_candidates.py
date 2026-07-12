import asyncio

from fastapi.testclient import TestClient

from app.main import create_app
from app.modules.search.api.dependencies import get_search_candidates_use_case
from app.shared.config.settings import get_settings


def _payload() -> dict[str, object]:
    return {
        "query": "tester1",
        "resource_types": ["users"],
        "candidate_limit_per_type": 5,
        "top_limit": 5,
        "cursors": {},
        "as_of": "2026-07-12T12:00:00Z",
        "filters": {},
    }


def test_internal_candidates_requires_nlp_service_token() -> None:
    client = TestClient(create_app())

    missing = client.post("/internal/search/candidates", json=_payload())
    wrong = client.post(
        "/internal/search/candidates",
        headers={"Authorization": "Bearer wrong"},
        json=_payload(),
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_internal_candidates_returns_only_ids_and_scores() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/internal/search/candidates",
        headers={"Authorization": "Bearer test-nlp-service-token"},
        json=_payload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"query", "sections", "pagination", "metadata"}
    assert payload["metadata"] == {
        "strategy": "hybrid_thresholded_candidates_v2",
        "failed_resources": {},
        "threshold_policy_version": "test-search-policy-v1",
    }
    candidate = payload["sections"]["users"][0]
    assert set(candidate) == {
        "id",
        "resource_type",
        "score",
        "semantic_score",
        "lexical_score",
    }
    assert "title" not in candidate
    assert "metadata" not in candidate


def test_internal_cursor_is_bound_to_as_of_and_policy_context() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer test-nlp-service-token"}
    first_payload = {
        "query": "lugar tranquilo",
        "resource_types": ["places"],
        "candidate_limit_per_type": 1,
        "top_limit": 1,
        "cursors": {},
        "as_of": "2026-07-12T12:00:00Z",
        "filters": {},
    }
    first = client.post(
        "/internal/search/candidates", headers=headers, json=first_payload
    )
    assert first.status_code == 200
    cursor = first.json()["pagination"]["places"]["next_cursor"]
    assert cursor

    changed = dict(first_payload)
    changed["as_of"] = "2026-07-12T12:01:00Z"
    changed["cursors"] = {"places": cursor}
    response = client.post(
        "/internal/search/candidates", headers=headers, json=changed
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "cursor belongs to another search context"


def test_internal_candidates_returns_503_when_request_budget_expires() -> None:
    class SlowUseCase:
        async def execute(self, **kwargs: object) -> None:
            await asyncio.sleep(0.05)

    app = create_app()
    app.dependency_overrides[get_search_candidates_use_case] = lambda: SlowUseCase()
    settings = get_settings()
    original_timeout = settings.request_timeout_seconds
    settings.request_timeout_seconds = 0.01  # type: ignore[assignment]
    try:
        response = TestClient(app).post(
            "/internal/search/candidates",
            headers={"Authorization": "Bearer test-nlp-service-token"},
            json=_payload(),
        )
    finally:
        settings.request_timeout_seconds = original_timeout

    assert response.status_code == 503
    assert response.json()["detail"] == "Internal search timed out"
