from dataclasses import dataclass


@dataclass(frozen=True)
class PostEmbedding:
    post_id: str
    vector: list[float]


@dataclass(frozen=True)
class ClusterAssignment:
    post_id: str
    cluster_id: int
    distance: float


@dataclass(frozen=True)
class KMeansResult:
    centroids: list[list[float]]
    assignments: list[ClusterAssignment]
    inertia: float
    silhouette_score: float | None
    parameters: dict[str, int | float | str]


@dataclass(frozen=True)
class ClusterRunResult:
    run_id: str
    k: int
    sample_size: int
    inertia: float
    silhouette_score: float | None
    status: str


@dataclass(frozen=True)
class ClusterStatus:
    active_run_id: str | None
    k: int | None
    sample_size: int | None
    silhouette_score: float | None
    activated_at: str | None


@dataclass(frozen=True)
class ClusterRunDetail:
    run_id: str
    status: str
    k: int
    sample_size: int
    inertia: float | None
    silhouette_score: float | None
    started_at: str
    completed_at: str | None
    activated_at: str | None
    error_message: str | None
