import pytest

from scripts import colab_initial_load_places


MODEL_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def restore_process_environment() -> None:
    original = dict(colab_initial_load_places.os.environ)
    yield
    colab_initial_load_places.os.environ.clear()
    colab_initial_load_places.os.environ.update(original)


def test_semantic_colab_profile_overrides_stale_legacy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLACES_EMBEDDING_MODEL", "owner/retriever")
    monkeypatch.setenv("PLACES_EMBEDDING_MODEL_REVISION", MODEL_SHA)
    monkeypatch.setenv(
        "PLACES_EMBEDDING_VERSION",
        f"places-e5-retriever-v1@{MODEL_SHA}",
    )
    monkeypatch.setenv("PLACES_EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setenv("PLACES_EMBEDDING_DIMENSION", "300")
    monkeypatch.setenv("PLACES_PGVECTOR_UPSERT_FUNCTION", "stale_writer")
    monkeypatch.setenv("PLACES_PGVECTOR_HASH_FUNCTION", "stale_hash")
    monkeypatch.setattr(
        colab_initial_load_places,
        "_read_required_secret",
        lambda _name: "writer-password",
    )

    colab_initial_load_places._configure_environment(semantic=True)

    assert colab_initial_load_places.os.environ[
        "PLACES_EMBEDDING_PROVIDER"
    ] == "sentence_transformer"
    assert colab_initial_load_places.os.environ[
        "PLACES_EMBEDDING_DIMENSION"
    ] == "768"
    assert colab_initial_load_places.os.environ[
        "PLACES_PGVECTOR_UPSERT_FUNCTION"
    ] == "upsert_place_embedding_semantic_v1"
    assert colab_initial_load_places.os.environ[
        "PLACES_PGVECTOR_HASH_FUNCTION"
    ] == "get_place_content_hashes_semantic_v1"
    assert colab_initial_load_places.os.environ[
        "PLACES_EMBEDDING_FIX_MISTRAL_REGEX"
    ] == "true"


def test_semantic_colab_profile_rejects_version_from_another_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLACES_EMBEDDING_MODEL", "owner/retriever")
    monkeypatch.setenv("PLACES_EMBEDDING_MODEL_REVISION", MODEL_SHA)
    monkeypatch.setenv(
        "PLACES_EMBEDDING_VERSION",
        "places-e5-retriever-v1@ffffffffffffffffffffffffffffffffffffffff",
    )

    with pytest.raises(RuntimeError, match="debe incluir el mismo SHA"):
        colab_initial_load_places._configure_environment(semantic=True)
