from app.shared.nlp.embeddings.mock import MockEmbeddingProvider


def test_mock_embedding_provider_is_deterministic() -> None:
    provider = MockEmbeddingProvider()

    first = provider.embed_text("lugares tranquilos para cenar")
    second = provider.embed_text("lugares tranquilos para cenar")

    assert first == second
    assert len(first) == provider.dimension
