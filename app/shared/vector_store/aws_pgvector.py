import json
import ssl
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from app.shared.config.settings import Settings
from app.shared.pgvector.sql import quote_identifier, vector_literal
from app.shared.vector_store.models import VectorMatch, VectorUpsertRecord


class AwsPgvectorClient:
    """PostgreSQL + pgvector access for RDS/Aurora using controlled SQL functions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dsn = _build_dsn(settings)
        self._ssl = _build_ssl_context(settings.pgvector_ssl_mode)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        connection = await asyncpg.connect(dsn=self._dsn, ssl=self._ssl)
        try:
            yield connection
        finally:
            await connection.close()

    async def match_places(
        self,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[VectorMatch]:
        return await self._match(
            function_name="match_places",
            embedding=embedding,
            filters=filters,
            limit=limit,
        )

    async def match_posts(
        self,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[VectorMatch]:
        return await self._match(
            function_name="match_posts",
            embedding=embedding,
            filters=filters,
            limit=limit,
        )

    async def fetch_place_content_hashes(self, ids: Iterable[str]) -> dict[str, str]:
        return await self._fetch_content_hashes(
            table_name=self._settings.pgvector_places_table,
            ids=ids,
        )

    async def fetch_post_content_hashes(self, ids: Iterable[str]) -> dict[str, str]:
        return await self._fetch_content_hashes(
            table_name=self._settings.pgvector_posts_table,
            ids=ids,
        )

    async def upsert_place_embeddings(
        self,
        records: list[VectorUpsertRecord],
    ) -> None:
        await self._upsert_records(
            function_name="upsert_place_embedding",
            records=records,
        )

    async def upsert_post_embeddings(
        self,
        records: list[VectorUpsertRecord],
    ) -> None:
        await self._upsert_records(
            function_name="upsert_post_embedding",
            records=records,
        )

    async def _match(
        self,
        function_name: str,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[VectorMatch]:
        query = f"SELECT * FROM {quote_identifier(function_name)}($1::vector, $2::integer, $3::jsonb)"
        async with self.connection() as connection:
            rows = await connection.fetch(
                query,
                vector_literal(embedding),
                limit,
                json.dumps(filters, ensure_ascii=False),
            )
        return [_row_to_vector_match(row) for row in rows]

    async def _fetch_content_hashes(
        self,
        table_name: str,
        ids: Iterable[str],
    ) -> dict[str, str]:
        id_list = list(ids)
        if not id_list:
            return {}

        table = quote_identifier(table_name)
        query = f"""
            SELECT external_id, content_hash
            FROM {table}
            WHERE external_id = ANY($1::text[])
        """
        async with self.connection() as connection:
            rows = await connection.fetch(query, id_list)
        return {
            str(row["external_id"]): str(row["content_hash"])
            for row in rows
            if row["content_hash"] is not None
        }

    async def _upsert_records(
        self,
        function_name: str,
        records: list[VectorUpsertRecord],
    ) -> None:
        if not records:
            return

        query = f"""
            SELECT {quote_identifier(function_name)}(
                $1::text,
                $2::text,
                $3::jsonb,
                $4::vector,
                $5::text,
                $6::text,
                $7::text,
                $8::boolean
            )
        """
        rows = [
            (
                record.id,
                record.document,
                json.dumps(record.metadata, ensure_ascii=False),
                vector_literal(record.embedding),
                record.content_hash,
                self._settings.embedding_model,
                self._settings.embedding_version,
                record.is_active,
            )
            for record in records
        ]
        async with self.connection() as connection:
            await connection.executemany(query, rows)


def _row_to_vector_match(row: Any) -> VectorMatch:
    metadata = _row_value(row, "metadata", default={}) or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    match_id = _row_value(row, "external_id") or _row_value(row, "id")
    score = _row_value(row, "score", default=0.0)
    return VectorMatch(
        id=str(match_id),
        score=float(score or 0.0),
        metadata=dict(metadata),
        document=_row_value(row, "document"),
    )


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _build_dsn(settings: Settings) -> str:
    if settings.pgvector_dsn:
        return settings.pgvector_dsn
    required = {
        "PGVECTOR_HOST": settings.pgvector_host,
        "PGVECTOR_DATABASE": settings.pgvector_database,
        "PGVECTOR_USER": settings.pgvector_user,
        "PGVECTOR_PASSWORD": settings.pgvector_password,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing pgvector settings: {', '.join(missing)}")
    return (
        f"postgresql://{settings.pgvector_user}:{settings.pgvector_password}"
        f"@{settings.pgvector_host}:{settings.pgvector_port}/{settings.pgvector_database}"
    )


def _build_ssl_context(mode: str | None) -> ssl.SSLContext | None:
    if mode != "require":
        raise RuntimeError("PGVECTOR_SSL_MODE must be require")
    context = ssl.create_default_context()
    return context
