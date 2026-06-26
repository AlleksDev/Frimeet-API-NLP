from app.shared.cache.memory import SimpleTTLCache
from app.shared.nlp.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: SimpleTTLCache,
    ) -> None:
        self._provider = provider
        self._cache = cache

    def embed_text(self, text: str) -> list[float]:
        cache_key = f"embedding:{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self._provider.embed_text(text)
        self._cache.set(cache_key, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]
