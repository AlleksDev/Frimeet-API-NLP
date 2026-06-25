import json
from typing import Any, Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.config.settings import Settings
from app.shared.pgvector.client import PgVectorConnectionFactory
from app.shared.pgvector.sql import quote_identifier, vector_literal


class PgVectorPlaceVectorRepository(PlaceVectorRepository):
    def __init__(
        self,
        connection_factory: PgVectorConnectionFactory,
        settings: Settings,
    ) -> None:
        self._connection_factory = connection_factory
        self._raw_table = settings.pgvector_places_table
        self._table = quote_identifier(settings.pgvector_places_table)
        self._dimension = settings.pgvector_embedding_dimension

    async def search(
        self,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        where_sql, params = _where_from_filters(filters.as_metadata_filter())
        embedding_value = vector_literal(embedding)
        query = f"""
            SELECT
                external_id,
                document,
                metadata,
                1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            WHERE embedding IS NOT NULL
            {where_sql}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params) + 2}
        """

        async with self._connection_factory.connection() as connection:
            rows = await connection.fetch(query, embedding_value, *params, limit)

        return [_row_to_place_candidate(row) for row in rows]

    async def ensure_schema(self) -> None:
        async with self._connection_factory.connection() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    external_id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({self._dimension}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_identifier(self._raw_table + "_metadata_gin_idx")}
                ON {self._table}
                USING GIN (metadata)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_identifier(self._raw_table + "_embedding_hnsw_idx")}
                ON {self._table}
                USING hnsw (embedding vector_cosine_ops)
                """
            )

    async def upsert_many(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        query = f"""
            INSERT INTO {self._table} (external_id, document, metadata, embedding, updated_at)
            VALUES ($1, $2, $3::jsonb, $4::vector, now())
            ON CONFLICT (external_id) DO UPDATE SET
                document = EXCLUDED.document,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        rows = [
            (
                place_id,
                document,
                json.dumps(metadata, ensure_ascii=False),
                vector_literal(embedding),
            )
            for place_id, document, metadata, embedding in zip(
                ids,
                documents,
                metadatas,
                embeddings,
            )
        ]
        async with self._connection_factory.connection() as connection:
            await connection.executemany(query, rows)


def _where_from_filters(metadata_filter: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    next_index = 2

    for key, value in metadata_filter.items():
        if value is None:
            continue
        if key == "occasion":
            clauses.append(f"metadata->>${next_index} ILIKE ${next_index + 1}")
            values.extend([key, f"%{value}%"])
        elif isinstance(value, bool):
            clauses.append(f"(metadata->>${next_index})::boolean = ${next_index + 1}")
            values.extend([key, value])
        else:
            clauses.append(f"lower(metadata->>${next_index}) = lower(${next_index + 1})")
            values.extend([key, str(value)])
        next_index += 2

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), values


def _row_to_place_candidate(row: Any) -> PlaceCandidate:
    metadata = dict(row["metadata"] or {})
    return PlaceCandidate(
        id=str(row["external_id"]),
        name=str(metadata.get("name") or row["external_id"]),
        score=float(row["score"] or 0.0),
        category=metadata.get("category"),
        city=metadata.get("city"),
        state=metadata.get("state"),
        price_range=metadata.get("price_range"),
        metadata=metadata,
    )
