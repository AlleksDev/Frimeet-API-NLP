from types import SimpleNamespace

from app.shared.vector_store.aws_pgvector import (
    _configured_read_contract_signatures,
    _read_contract_is_ready,
)


def _functions(search_v2: bool) -> dict[str, dict[str, object]]:
    return {
        "match_places": {"exists": True, "executable": True},
        "match_posts": {"exists": True, "executable": True},
        "search_resource_embeddings": {
            "exists": True,
            "executable": True,
            "candidate_filters_v2": search_v2,
        },
        "get_post_feed_features": {"exists": True, "executable": True},
    }


def test_readiness_rejects_legacy_search_function() -> None:
    assert _read_contract_is_ready(True, _functions(search_v2=False)) is False


def test_readiness_accepts_search_candidate_v2_contract() -> None:
    assert _read_contract_is_ready(True, _functions(search_v2=True)) is True


def test_readiness_includes_configured_places_semantic_contract() -> None:
    signatures = _configured_read_contract_signatures(
        SimpleNamespace(
            places_pgvector_match_function="match_places_semantic_v1",
            places_pgvector_hybrid_function="search_places_semantic_v1",
        )
    )

    assert signatures["match_places_semantic_v1"] == (
        "match_places_semantic_v1(vector, integer, jsonb)"
    )
    assert signatures["search_places_semantic_v1"] == (
        "search_places_semantic_v1(text, vector, integer, jsonb)"
    )
