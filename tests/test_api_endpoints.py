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
    assert payload["metrics"]["engine"] == "tfidf"
    assert payload["metrics"]["candidate_retrieval"] == "embeddings"
    assert payload["metrics"]["score_metric"] == "cosine_similarity"
    assert payload["metrics"]["field_weights"]["tags"] == 6
    assert payload["metrics"]["returned_count"] == len(payload["places"])
    assert payload["metrics"]["max_score"] >= payload["metrics"]["mean_score"]


def test_places_search_metrics_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/search/metrics",
        json={
            "k": 3,
            "cases": [
                {
                    "query": "atardecer fotos paseo",
                    "relevance": {"place_2": 3},
                    "filters": {"is_active": True},
                },
                {
                    "query": "cafe tranquilo con postres",
                    "relevance": {"place_1": 3},
                    "filters": {"is_active": True},
                },
                {
                    "query": "comida regional restaurante",
                    "relevance": {"place_3": 3},
                    "filters": {"is_active": True},
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "pgvector_candidates_plus_tfidf_cosine"
    assert payload["query_count"] == 3
    assert payload["aggregate"]["recall_at_k"] == 1.0
    assert payload["aggregate"]["mrr"] == 1.0
    assert payload["aggregate"]["map"] == 1.0
    assert payload["aggregate"]["ndcg_at_k"] == 1.0


def test_places_search_metrics_rejects_invalid_relevance_grade() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/places/search/metrics",
        json={
            "cases": [
                {
                    "query": "cafe",
                    "relevance": {"place_1": 4},
                }
            ]
        },
    )

    assert response.status_code == 422


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


def test_places_recommendations_returns_llm_message_and_tfidf_metadata() -> None:
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
    assert payload["metadata"]["ranking"] == "tfidf_cosine"
    assert payload["metadata"]["used_llm"] is True


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
