import pytest
from pydantic import ValidationError

from app.shared.config.settings import Settings


def test_settings_accepts_nlp_service_token_as_canonical_alias() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        NLP_SERVICE_TOKEN="canonical-token",
    )
    assert settings.post_feed_internal_token == "canonical-token"


def test_production_rejects_mock_vector_store_for_feed() -> None:
    with pytest.raises(ValidationError, match="VECTOR_STORE_PROVIDER=aws_pgvector"):
        Settings(
            _env_file=None,
            ENV="production",
            NLP_SERVICE_TOKEN="nlp-token",
            MAIN_API_INTERNAL_TOKEN="main-token",
            VECTOR_STORE_PROVIDER="mock",
        )


def test_search_threshold_overrides_accept_complete_resource_policy() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        NLP_SERVICE_TOKEN="nlp-token",
        GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON={
            "posts": {"semantic_min": 0.42, "lexical_min": 0.08}
        },
    )

    assert settings.global_search_resource_thresholds["posts"] == {
        "semantic_min": 0.42,
        "lexical_min": 0.08,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"posts": {"semantic_min": 0.42}},
        {"unknown": {"semantic_min": 0.42, "lexical_min": 0.08}},
        {"posts": {"semantic_min": 1.1, "lexical_min": 0.08}},
    ],
)
def test_search_threshold_overrides_reject_invalid_policy(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ENV="local",
            NLP_SERVICE_TOKEN="nlp-token",
            GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON=overrides,
        )
