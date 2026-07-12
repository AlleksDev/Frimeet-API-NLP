from typing import Protocol


class QueryEmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]:
        """Build one embedding for a normalized search query."""
