from collections.abc import Callable

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
        return self.embed_document(text)

    def embed_query(self, text: str) -> list[float]:
        return self._get_or_create("query", text, self._provider.embed_query)

    def embed_document(self, text: str) -> list[float]:
        return self._get_or_create("document", text, self._provider.embed_document)

    def _get_or_create(
        self,
        kind: str,
        text: str,
        embed: Callable[[str], list[float]],
    ) -> list[float]:
        cache_key = f"embedding:{kind}:{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = embed(text)
        self._cache.set(cache_key, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]
