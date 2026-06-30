from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for many texts."""

    def embed_query(self, text: str) -> list[float]:
        """Generate a search-query embedding.

        Symmetric encoders such as FastText can use the same representation for
        queries and documents. Retrieval encoders can override this method.
        """
        return self.embed_text(text)

    def embed_document(self, text: str) -> list[float]:
        """Generate an embedding for a document stored in the vector index."""
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate document embeddings in a batch."""
        return self.embed_batch(texts)
