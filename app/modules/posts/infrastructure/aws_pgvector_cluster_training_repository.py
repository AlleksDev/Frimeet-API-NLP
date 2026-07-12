import json
from collections import defaultdict
from uuid import uuid4

from app.modules.posts.domain.clustering import (
    ClusterRunDetail,
    ClusterStatus,
    KMeansResult,
    PostEmbedding,
)
from app.modules.posts.domain.ports.cluster_operations_repository import ClusterOperationsRepository
from app.modules.posts.domain.ports.cluster_training_repository import ClusterTrainingRepository
from app.shared.config.settings import Settings
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.sql import vector_literal


class AwsPgvectorClusterTrainingRepository(
    ClusterTrainingRepository, ClusterOperationsRepository
):
    def __init__(self, client: AwsPgvectorClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def list_active_embeddings(self, lookback_days: int) -> list[PostEmbedding]:
        async with self._client.connection() as connection:
            rows = await connection.fetch(
                """
                SELECT post_id, embedding::text
                FROM list_post_embeddings_for_clustering(
                    $1::integer, $2::text, $3::text
                )
                """,
                lookback_days,
                self._settings.embedding_model,
                self._settings.embedding_version,
            )
        return [
            PostEmbedding(str(row["post_id"]), _parse_vector(row["embedding"]))
            for row in rows
        ]

    async def create_run(
        self,
        planned_k: int,
        sample_size: int,
        algorithm_version: str,
        parameters: dict[str, int | float | str],
    ) -> str:
        run_id = str(uuid4())
        async with self._client.connection() as connection:
            await connection.execute(
                """
                SELECT create_post_cluster_run(
                    $1::uuid, $2::text, $3::text, $4::text, $5::integer,
                    $6::integer, $7::integer, $8::jsonb
                )
                """,
                run_id,
                algorithm_version,
                self._settings.embedding_model,
                self._settings.embedding_version,
                self._settings.embedding_dimension,
                planned_k,
                sample_size,
                json.dumps(parameters),
            )
        return run_id

    async def save_validated_run(
        self, run_id: str, k: int, result: KMeansResult
    ) -> None:
        representatives: dict[int, list[str]] = defaultdict(list)
        for item in sorted(result.assignments, key=lambda value: value.distance):
            if len(representatives[item.cluster_id]) < 5:
                representatives[item.cluster_id].append(item.post_id)
        clusters = [
            {
                "cluster_id": cluster_id,
                "centroid": vector_literal(centroid),
                "size": sum(
                    1 for item in result.assignments if item.cluster_id == cluster_id
                ),
                "representative_post_ids": representatives[cluster_id],
            }
            for cluster_id, centroid in enumerate(result.centroids)
        ]
        memberships = [
            {
                "post_id": item.post_id,
                "cluster_id": item.cluster_id,
                "distance": item.distance,
            }
            for item in result.assignments
        ]
        async with self._client.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    SELECT validate_post_cluster_run(
                        $1::uuid, $2::integer, $3::double precision,
                        $4::double precision, $5::jsonb
                    )
                    """,
                    run_id,
                    k,
                    result.inertia,
                    result.silhouette_score,
                    json.dumps(result.parameters),
                )
                await connection.execute(
                    "SELECT replace_post_cluster_artifacts($1::uuid, $2::jsonb, $3::jsonb)",
                    run_id,
                    json.dumps(clusters),
                    json.dumps(memberships),
                )

    async def mark_run_failed(self, run_id: str, error_message: str) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT fail_post_cluster_run($1::uuid, $2::text)",
                run_id,
                error_message[:4000],
            )

    async def get_status(self) -> ClusterStatus:
        async with self._client.connection() as connection:
            row = await connection.fetchrow("SELECT * FROM get_post_cluster_status()")
        if row is None or row["active_run_id"] is None:
            return ClusterStatus(None, None, None, None, None)
        return ClusterStatus(
            active_run_id=str(row["active_run_id"]),
            k=int(row["k"]),
            sample_size=int(row["sample_size"]),
            silhouette_score=(
                float(row["silhouette_score"])
                if row["silhouette_score"] is not None
                else None
            ),
            activated_at=str(row["activated_at"]) if row["activated_at"] else None,
        )

    async def get_run(self, run_id: str) -> ClusterRunDetail | None:
        async with self._client.connection() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM get_post_cluster_run($1::uuid)", run_id
            )
        if row is None:
            return None
        return ClusterRunDetail(
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            k=int(row["k"]),
            sample_size=int(row["sample_size"]),
            inertia=float(row["inertia"]) if row["inertia"] is not None else None,
            silhouette_score=(
                float(row["silhouette_score"])
                if row["silhouette_score"] is not None
                else None
            ),
            started_at=str(row["started_at"]),
            completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            activated_at=str(row["activated_at"]) if row["activated_at"] else None,
            error_message=str(row["error_message"]) if row["error_message"] else None,
        )

    async def activate_run(self, run_id: str) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT activate_post_cluster_run($1::uuid)", run_id
            )


def _parse_vector(value: str) -> list[float]:
    return [float(item) for item in value.strip("[]").split(",") if item]
