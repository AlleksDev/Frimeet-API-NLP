from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.cached import CachedEmbeddingProvider
from app.shared.cache.memory import SimpleTTLCache


def test_mock_embedding_provider_is_deterministic() -> None:
    provider = MockEmbeddingProvider()

    first = provider.embed_text("lugares tranquilos para cenar")
    second = provider.embed_text("lugares tranquilos para cenar")

    assert first == second
    assert len(first) == provider.dimension


class RecordingBatchProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        raise AssertionError("batch cache should use embed_batch for misses")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text))] for text in texts]


def test_cached_provider_batches_unique_misses_and_preserves_order() -> None:
    inner = RecordingBatchProvider()
    provider = CachedEmbeddingProvider(inner, SimpleTTLCache())

    first = provider.embed_batch(["donas", "cafe", "donas"])
    second = provider.embed_batch(["cafe", "parque"])

    assert first == [[5.0], [4.0], [5.0]]
    assert second == [[4.0], [6.0]]
    assert inner.batches == [["donas", "cafe"], ["parque"]]
