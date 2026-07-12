from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from app.modules.posts.domain.clustering import ClusterAssignment, KMeansResult, PostEmbedding
from app.modules.posts.domain.ports.clusterer import PostClusterer


class MiniBatchKMeansPostClusterer(PostClusterer):
    """Production adapter: deterministic MiniBatchKMeans over L2-normalized vectors."""

    def __init__(
        self,
        random_state: int = 42,
        batch_size: int = 1024,
        max_iterations: int = 100,
        n_init: int = 10,
        silhouette_sample_size: int = 5_000,
    ) -> None:
        self._random_state = random_state
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._n_init = n_init
        self._silhouette_sample_size = silhouette_sample_size

    def fit(self, embeddings: list[PostEmbedding], k: int) -> KMeansResult:
        if k < 2 or k >= len(embeddings):
            raise ValueError("k debe ser mayor a 1 y menor que el numero de posts")
        matrix = _normalized_matrix(embeddings)
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self._random_state,
            batch_size=min(self._batch_size, len(embeddings)),
            max_iter=self._max_iterations,
            n_init=self._n_init,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(matrix)
        centroids = _normalize_rows(model.cluster_centers_)
        distances = np.linalg.norm(matrix - centroids[labels], axis=1)
        score = _silhouette(matrix, labels, self._silhouette_sample_size, self._random_state)
        parameters: dict[str, Any] = {
            "random_state": self._random_state,
            "batch_size": min(self._batch_size, len(embeddings)),
            "max_iterations": self._max_iterations,
            "n_init": self._n_init,
        }
        return KMeansResult(
            centroids=centroids.tolist(),
            assignments=[
                ClusterAssignment(item.post_id, int(labels[index]), float(distances[index]))
                for index, item in enumerate(embeddings)
            ],
            inertia=float(model.inertia_),
            silhouette_score=score,
            parameters=parameters,
        )


def _normalized_matrix(embeddings: list[PostEmbedding]) -> np.ndarray:
    dimensions = {len(item.vector) for item in embeddings}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
        raise ValueError("todos los embeddings deben tener la misma dimension no vacia")
    matrix = np.asarray([item.vector for item in embeddings], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("los embeddings no pueden contener NaN o infinito")
    return _normalize_rows(matrix)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("los embeddings no pueden ser vectores cero")
    return matrix / norms


def _silhouette(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(matrix):
        return None
    return float(
        silhouette_score(
            matrix,
            labels,
            metric="euclidean",
            sample_size=min(sample_size, len(matrix)),
            random_state=random_state,
        )
    )


# Backward-compatible name for existing imports.
KMeansPostClusterer = MiniBatchKMeansPostClusterer
