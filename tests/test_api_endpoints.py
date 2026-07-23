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
    assert payload["metrics"]["engine"] == "mock-place-embedding"
    assert payload["metrics"]["candidate_retrieval"] == "mock_embeddings"
    assert payload["metrics"]["score_metric"] == "cosine_similarity"
    assert payload["metrics"]["ranking_parameters"] == {"dimension": 16.0}
    assert payload["metrics"]["field_weights"] == {
        "tags": 1,
        "category": 1,
        "description": 1,
        "name": 1,
        "attributes": 1,
        "entertainment": 1,
        "contained_items": 1,
        "menu": 1,
    }
    assert payload["metrics"]["returned_count"] == len(payload["places"])
    assert payload["metrics"]["max_score"] >= payload["metrics"]["mean_score"]


def test_places_search_metrics_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/places/search/metrics?k=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "bm25"
    assert payload["benchmark"] == "built_in_places_v3_bm25"
    assert payload["qrels_source"] == "predefined_graded_qrels"
    assert payload["query_count"] == 10
    assert payload["metric_definitions"]["precision_at_k"]["label"] == "Precision@3"
    assert payload["metric_definitions"]["recall_at_k"]["label"] == "Recall@3"
    assert payload["metric_definitions"]["mrr"]["label"] == "MRR"
    assert payload["metric_definitions"]["map"]["label"] == "MAP"
    assert payload["metric_definitions"]["ndcg_at_k"]["label"] == "nDCG@3"
    assert payload["recommended_metric"]["key"] == "ndcg_at_k"
    assert payload["recommended_metric"]["label"] == "nDCG@3"
    assert payload["recommended_metric"]["value"] == payload["aggregate"]["ndcg_at_k"]
    assert all(
        0.0 <= payload["aggregate"][metric] <= 1.0
        for metric in ["precision_at_k", "recall_at_k", "mrr", "map", "ndcg_at_k"]
    )


def test_places_search_metrics_post_requires_no_body() -> None:
    client = TestClient(create_app())

    response = client.post("/places/search/metrics?k=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["k"] == 2
    assert payload["query_count"] == 10


def test_places_chat_endpoint_returns_trace_and_structured_places() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/chat",
        json={
            "message": "cafe tranquilo postres platica",
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


def test_places_chat_opt_in_uses_semantic_conversation_contract() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/chat",
        json={
            "conversation_id": "3a4723f6-260d-4c11-b9d6-089f07a4f338",
            "turn": 1,
            "message": "una cafeteria tranquila",
            "conversation_state": {
                "city": "Tuxtla Gutierrez",
                "taxonomy_version": "places-taxonomy-v1",
            },
            "user_location": {"lat": 16.7531, "lng": -93.1156},
            "filters": {"is_active": True},
            "candidate_limit": 5,
            "limit": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "recommendations"
    assert payload["state_patch"]["target_category"] == "cafe"
    assert payload["state_patch"]["taxonomy_version"] == "places-taxonomy-v3"
    assert payload["location_directive"]["source"] == "user_current"
    assert payload["uncertainty"]["decision"] == "auto"
    assert payload["metadata"]["pipeline"] == "places-chat-semantic-v2"
    assert len(payload["places"]) == 1


def test_places_chat_opt_in_rejects_incompatible_conversation_state() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/chat",
        json={
            "conversation_id": "3a4723f6-260d-4c11-b9d6-089f07a4f338",
            "message": "donas",
            "conversation_state": {"taxonomy_version": "obsolete-v0"},
            "user_location": {"lat": 16.7531, "lng": -93.1156},
        },
    )

    assert response.status_code == 409


def test_places_chat_opt_in_maps_stale_clarification_to_conflict() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/chat",
        json={
            "conversation_id": "3a4723f6-260d-4c11-b9d6-089f07a4f338",
            "message": "donas",
            "conversation_state": {"taxonomy_version": "places-taxonomy-v1"},
            "clarification_choice": {
                "clarification_id": "00000000-0000-4000-8000-000000000000",
                "option_id": "bakery",
            },
            "user_location": {"lat": 16.7531, "lng": -93.1156},
        },
    )

    assert response.status_code == 409


def test_places_recommendations_returns_llm_message_and_semantic_metadata() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/recommendations",
        json={
            "query": "quiero ver el atardecer y tomar fotos",
            "city": "Tuxtla Gutierrez",
            "filters": {"is_active": True},
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]
    assert payload["places"]
    assert payload["metrics"]["engine"] == "mock-place-embedding"
    assert payload["metrics"]["score_metric"] == "cosine_similarity"
    assert payload["metrics"]["returned_count"] == len(payload["places"])
    assert payload["metrics"]["candidate_retrieval"] == "mock_embeddings"
    assert payload["metrics"]["query_token_count"] > 0
    assert payload["metrics"]["matched_query_token_count"] > 0
    assert payload["metrics"]["scope"] == "current_query"
    assert payload["metrics"]["ground_truth_available"] is False
    assert "evaluation_metrics" not in payload
    assert payload["metadata"]["ranking"] == "mock-place-embedding"
    assert payload["metadata"]["response_mode"] == "confident"
    assert payload["metadata"]["used_llm"] is True


def test_places_recommendations_returns_no_places_without_candidates() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/recommendations",
        json={
            "query": "xqzv blorf 998zz",
            "city": "Ciudad inexistente",
            "filters": {"is_active": True},
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["places"] == []
    assert payload["metrics"]["max_score"] == 0.0
    assert payload["metrics"]["returned_count"] == 0
    assert payload["metrics"]["match_quality"] == "no_match"
    assert payload["metadata"]["response_mode"] == "no_match"
    assert payload["message"]


def test_places_search_rejects_incomplete_coordinates() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/search",
        json={
            "query": "parque cercano",
            "lat": 16.7531,
            "limit": 5,
        },
    )

    assert response.status_code == 422


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

    assert response.status_code == 404
