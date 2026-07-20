import json

import pytest
from pydantic import ValidationError

from app.jobs.sync_place_embeddings import _validate_backfill_configuration
from app.modules.places.api import dependencies as place_dependencies
from app.modules.places.infrastructure.bert_intent_extractor import (
    BertPlaceIntentExtractor,
)
from app.modules.places.infrastructure.place_category_catalog import (
    load_place_category_concepts,
)
from app.shared.config.settings import Settings
from app.shared.nlp.embeddings.factory import create_place_embedding_provider
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


def test_places_embedding_provider_is_independent_from_global_dimension() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        EMBEDDING_PROVIDER="mock",
        EMBEDDING_DIMENSION=16,
        PLACES_EMBEDDING_PROVIDER="mock",
        PLACES_EMBEDDING_DIMENSION=384,
        PLACES_EMBEDDING_MODEL="places-test",
    )

    provider = create_place_embedding_provider(settings)

    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimension == 384
    assert settings.embedding_dimension == 16


def test_blank_optional_places_runtime_values_are_normalized_to_none() -> None:
    settings = Settings(
        _env_file=None,
        PLACES_EMBEDDING_DEVICE="   ",
        PLACES_EMBEDDING_MODEL_REVISION="   ",
        PLACES_CATEGORY_CATALOG_PATH="",
        PLACES_PGVECTOR_HYBRID_FUNCTION=" ",
    )

    assert settings.places_embedding_device is None
    assert settings.places_embedding_model_revision is None
    assert settings.places_category_catalog_path is None
    assert settings.places_pgvector_hybrid_function is None


def test_sentence_transformer_factory_configures_query_and_passage_prefixes() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        PLACES_EMBEDDING_PROVIDER="sentence_transformer",
        PLACES_EMBEDDING_DIMENSION=768,
        PLACES_EMBEDDING_MODEL="intfloat/multilingual-e5-base",
        PLACES_EMBEDDING_MODEL_REVISION=(
            "  0123456789abcdef0123456789abcdef01234567  "
        ),
        PLACES_EMBEDDING_QUERY_PREFIX="query: ",
        PLACES_EMBEDDING_PASSAGE_PREFIX="passage: ",
    )

    query = create_place_embedding_provider(settings, text_role="query")
    passage = create_place_embedding_provider(settings, text_role="passage")

    assert isinstance(query, SentenceTransformerEmbeddingProvider)
    assert isinstance(passage, SentenceTransformerEmbeddingProvider)
    assert query.text_prefix == "query: "
    assert passage.text_prefix == "passage: "
    assert query.model_revision == "0123456789abcdef0123456789abcdef01234567"
    assert passage.model_revision == "0123456789abcdef0123456789abcdef01234567"
    assert query.fix_mistral_regex is True
    assert passage.fix_mistral_regex is True
    assert query.is_loaded is False
    assert passage.is_loaded is False


def test_quoted_e5_prefixes_preserve_the_significant_space(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'PLACES_EMBEDDING_QUERY_PREFIX="query: "\n'
        'PLACES_EMBEDDING_PASSAGE_PREFIX="passage: "\n',
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.places_embedding_query_prefix == "query: "
    assert settings.places_embedding_passage_prefix == "passage: "


def test_e5_prefixes_recover_space_trimmed_by_deployment_ui() -> None:
    settings = Settings(
        _env_file=None,
        PLACES_EMBEDDING_QUERY_PREFIX="query:",
        PLACES_EMBEDDING_PASSAGE_PREFIX="passage:",
    )

    assert settings.places_embedding_query_prefix == "query: "
    assert settings.places_embedding_passage_prefix == "passage: "


def test_places_embedding_provider_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError, match="PLACES_EMBEDDING_PROVIDER"):
        Settings(
            _env_file=None,
            ENV="local",
            PLACES_EMBEDDING_PROVIDER="closed_taxonomy_magic",
        )


def test_semantic_backfill_requires_a_coherent_768d_writer_profile() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        PLACES_EMBEDDING_PROVIDER="sentence_transformer",
        PLACES_EMBEDDING_DIMENSION=768,
        PLACES_EMBEDDING_MODEL="owner/retriever",
        PLACES_EMBEDDING_MODEL_REVISION=(
            "0123456789abcdef0123456789abcdef01234567"
        ),
        PLACES_EMBEDDING_PASSAGE_PREFIX="passage:",
        PLACES_PGVECTOR_UPSERT_FUNCTION="upsert_place_embedding_semantic_v1",
        PLACES_PGVECTOR_HASH_FUNCTION=(
            "get_place_content_hashes_semantic_v1"
        ),
    )

    _validate_backfill_configuration(settings)

    settings.places_pgvector_hash_function = "get_place_content_hashes"
    with pytest.raises(RuntimeError, match="requires both writer functions"):
        _validate_backfill_configuration(settings)


def test_768d_retriever_cannot_write_to_legacy_places_table() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        PLACES_EMBEDDING_PROVIDER="sentence_transformer",
        PLACES_EMBEDDING_DIMENSION=768,
        PLACES_EMBEDDING_MODEL="owner/retriever",
    )

    with pytest.raises(RuntimeError, match="legacy 300d"):
        _validate_backfill_configuration(settings)


def test_production_sentence_transformer_requires_full_commit_sha() -> None:
    common = {
        "_env_file": None,
        "ENV": "production",
        "NLP_SERVICE_TOKEN": "service-token",
        "MAIN_API_INTERNAL_TOKEN": "main-api-token",
        "VECTOR_STORE_PROVIDER": "aws_pgvector",
        "PLACES_EMBEDDING_PROVIDER": "sentence_transformer",
        "PLACES_EMBEDDING_DIMENSION": 768,
        "PLACES_EMBEDDING_MODEL": "owner/retriever",
    }

    with pytest.raises(
        ValidationError,
        match="PLACES_EMBEDDING_MODEL_REVISION",
    ):
        Settings(**common, PLACES_EMBEDDING_MODEL_REVISION="main")

    settings = Settings(
        **common,
        PLACES_EMBEDDING_MODEL_REVISION=(
            "0123456789abcdef0123456789abcdef01234567"
        ),
    )

    assert settings.places_embedding_model_revision == (
        "0123456789abcdef0123456789abcdef01234567"
    )


def test_external_place_concept_catalog_is_data_driven(tmp_path) -> None:
    path = tmp_path / "concepts.json"
    path.write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "id": "donut_shop",
                        "label": "Donas",
                        "description": "Lugar especializado en donas artesanales",
                        "examples": ["algo dulce glaseado"],
                        "storage_values": ["bakery", "dessert"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    concepts = load_place_category_concepts(str(path))

    assert [concept.id for concept in concepts] == ["donut_shop"]
    assert concepts[0].storage_values == ("bakery", "dessert")


def test_place_chat_intent_provider_defaults_to_deterministic() -> None:
    settings = Settings(_env_file=None, ENV="local")

    assert settings.places_chat_intent_provider == "deterministic"
    assert settings.places_chat_bert_model_path is None


def test_place_chat_bert_provider_is_explicit_and_validated() -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        PLACES_CHAT_INTENT_PROVIDER="BERT",
        PLACES_CHAT_BERT_MODEL_PATH="models/places-intent",
        PLACES_CHAT_BERT_MODEL_VERSION="intent-v3",
        PLACES_CHAT_BERT_DEVICE="cpu",
        PLACES_CHAT_BERT_MIN_TOKEN_CONFIDENCE=0.72,
    )

    assert settings.places_chat_intent_provider == "bert"
    assert settings.places_chat_bert_model_path == "models/places-intent"
    assert settings.places_chat_bert_model_version == "intent-v3"
    assert settings.places_chat_bert_device == "cpu"
    assert settings.places_chat_bert_min_token_confidence == pytest.approx(0.72)

    with pytest.raises(ValidationError, match="PLACES_CHAT_BERT_MODEL_PATH"):
        Settings(
            _env_file=None,
            ENV="local",
            PLACES_CHAT_INTENT_PROVIDER="bert",
        )


def test_place_chat_dependency_builds_bert_extractor_without_loading_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        ENV="local",
        PLACES_EMBEDDING_PROVIDER="mock",
        PLACES_CHAT_INTENT_PROVIDER="bert",
        PLACES_CHAT_BERT_MODEL_PATH="models/places-intent",
        PLACES_CHAT_BERT_MODEL_VERSION="intent-v3",
        PLACES_CHAT_BERT_DEVICE="cpu",
    )
    monkeypatch.setattr(place_dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(
        place_dependencies,
        "load_place_category_concepts",
        lambda *_args, **_kwargs: (),
    )
    place_dependencies.get_place_chat_intent_parser.cache_clear()
    try:
        parser = place_dependencies.get_place_chat_intent_parser()
        extractor = parser._contextual_extractor

        assert isinstance(extractor, BertPlaceIntentExtractor)
        assert extractor.model_name == "models/places-intent"
        assert extractor.model_version == "intent-v3"
        assert extractor.is_loaded is False
    finally:
        place_dependencies.get_place_chat_intent_parser.cache_clear()
