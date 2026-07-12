from typing import Protocol

from app.modules.posts.domain.clustering import KMeansResult, PostEmbedding


class PostClusterer(Protocol):
    def fit(self, embeddings: list[PostEmbedding], k: int) -> KMeansResult:
        """Cluster normalized post embeddings using K-Means."""
