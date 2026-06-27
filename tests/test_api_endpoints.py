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

    response = client.get("/places/search/metrics?k=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "tfidf_cosine"
    assert payload["benchmark"] == "built_in_places_v1"
    assert payload["qrels_source"] == "predefined_graded_qrels"
    assert payload["query_count"] == 5
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
    assert payload["query_count"] == 5


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
    assert payload["metrics"]["engine"] == "tfidf"
    assert payload["metrics"]["score_metric"] == "cosine_similarity"
    assert payload["metrics"]["returned_count"] == len(payload["places"])
    assert payload["evaluation_metrics"]["benchmark"] == "built_in_places_v1"
    assert payload["evaluation_metrics"]["query_count"] == 5
    assert payload["evaluation_metrics"]["recommended_metric"]["key"] == "ndcg_at_k"
    assert all(
        metric in payload["evaluation_metrics"]["aggregate"]
        for metric in ["precision_at_k", "recall_at_k", "mrr", "map", "ndcg_at_k"]
    )
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
