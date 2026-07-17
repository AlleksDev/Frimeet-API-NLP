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
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_positions: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            cache_key = f"embedding:{text}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[index] = cached
            else:
                missing_positions.setdefault(text, []).append(index)

        missing_texts = list(missing_positions)
        if missing_texts:
            generated = self._provider.embed_batch(missing_texts)
            if len(generated) != len(missing_texts):
                raise ValueError(
                    "Embedding provider returned an unexpected batch size: "
                    f"returned={len(generated)}, expected={len(missing_texts)}"
                )
            for text, embedding in zip(missing_texts, generated):
                self._cache.set(f"embedding:{text}", embedding)
                for index in missing_positions[text]:
                    results[index] = embedding

        if any(result is None for result in results):
            raise RuntimeError("Embedding batch cache left unresolved positions")
        return [result for result in results if result is not None]
