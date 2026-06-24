import hashlib
import math

from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        normalized = prepare_for_embedding(text)
        if not normalized:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for token in normalized.split():
            token_vector = self._token_vector(token)
            vector = [left + right for left, right in zip(vector, token_vector)]
        return _l2_normalize(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def _token_vector(self, token: str) -> list[float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return [
            (digest[index % len(digest)] / 255.0) * 2.0 - 1.0
            for index in range(self.dimension)
        ]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
