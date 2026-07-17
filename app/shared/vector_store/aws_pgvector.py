import json
import ssl
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from app.shared.config.settings import Settings
from app.shared.errors.exceptions import AppError
from app.shared.vector_store.models import VectorMatch, VectorUpsertRecord
from app.shared.vector_store.sql import quote_identifier, vector_literal


READ_CONTRACT_SIGNATURES = {
    "match_places": "match_places(vector, integer, jsonb)",
    "match_posts": "match_posts(vector, integer, jsonb)",
    "search_resource_embeddings": (
        "search_resource_embeddings(text, text, vector, integer, jsonb)"
    ),
    "get_post_feed_features": "get_post_feed_features(text, text[])",
}

SEARCH_CANDIDATE_V2_MARKERS = (
    "event_active_at",
    "min_semantic_score",
    "min_lexical_score",
    "try_parse_timestamptz",
)

SEARCH_RESOURCE_TYPES = frozenset({"places", "posts", "users", "clubs", "groups", "events"})


class AwsPgvectorClient:
    """PostgreSQL + pgvector access for RDS/Aurora using controlled SQL functions."""

    def __init__(self, settings: Settings, role: str = "reader") -> None:
        self._settings = settings
        self._role = role
        self._connection_kwargs = _build_connection_kwargs(settings, role)
        self._ssl = _build_ssl_context(settings.pgvector_ssl_mode)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        connection = await asyncpg.connect(**self._connection_kwargs, ssl=self._ssl)
        try:
            yield connection
        finally:
            await connection.close()

    async def match_places(
        self,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
        function_name: str = "match_places",
    ) -> list[VectorMatch]:
        return await self._match(
            function_name=function_name,
            embedding=embedding,
            filters=filters,
            limit=limit,
        )

    async def search_places_hybrid(
        self,
        query_text: str,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
        function_name: str,
    ) -> list[VectorMatch]:
        """Search a versioned Places index using independent dense and lexical pools."""

        query = (
            f"SELECT * FROM {quote_identifier(function_name)}("
            "$1::text, $2::vector, $3::integer, $4::jsonb)"
        )
        try:
            async with self.connection() as connection:
                rows = await connection.fetch(
                    query,
                    query_text,
                    vector_literal(embedding),
                    limit,
                    json.dumps(filters, ensure_ascii=False),
                )
        except asyncpg.exceptions.UndefinedFunctionError as exc:
            raise AppError(
                "Places hybrid SQL contract is missing. Run the versioned "
                "Places semantic embedding migration and grant EXECUTE to the "
                "reader role.",
                code="pgvector_places_hybrid_contract_missing",
                status_code=503,
            ) from exc
        return [_row_to_vector_match(row) for row in rows]

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

    async def search_resource_embeddings(
        self,
        resource_type: str,
        query_text: str,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[VectorMatch]:
        _validate_resource_type(resource_type)
        query = (
            "SELECT * FROM search_resource_embeddings("
            "$1::text, $2::text, $3::vector, $4::integer, $5::jsonb)"
        )
        try:
            async with self.connection() as connection:
                rows = await connection.fetch(
                    query,
                    resource_type,
                    query_text,
                    vector_literal(embedding),
                    limit,
                    json.dumps(filters, ensure_ascii=False),
                )
        except asyncpg.exceptions.UndefinedFunctionError as exc:
            raise AppError(
                "Hybrid search SQL contract is missing. Run "
                "sql/aws_pgvector_contract.sql and grant EXECUTE to nlp_reader.",
                code="pgvector_hybrid_contract_missing",
                status_code=503,
            ) from exc
        return [_row_to_vector_match(row) for row in rows]

    async def check_read_contract(self) -> dict[str, Any]:
        """Check that read-only pgvector functions are visible and executable."""
        signatures = _configured_read_contract_signatures(self._settings)
        try:
            async with self.connection() as connection:
                vector_row = await connection.fetchrow(
                    "SELECT to_regtype('vector') IS NOT NULL AS available"
                )
                vector_available = bool(vector_row["available"]) if vector_row else False
                functions: dict[str, dict[str, Any]] = {
                    function_name: {
                        "signature": signature,
                        "exists": False,
                        "executable": False,
                    }
                    for function_name, signature in signatures.items()
                }
                if vector_available:
                    for function_name, signature in signatures.items():
                        row = await connection.fetchrow(
                            """
                            SELECT
                                to_regprocedure($1) IS NOT NULL AS exists,
                                COALESCE(
                                    has_function_privilege(to_regprocedure($1), 'EXECUTE'),
                                    false
                                ) AS executable
                            """,
                            signature,
                        )
                        functions[function_name].update(
                            {
                                "exists": bool(row["exists"]) if row else False,
                                "executable": bool(row["executable"]) if row else False,
                            }
                        )
                        if function_name == "search_resource_embeddings":
                            definition_row = await connection.fetchrow(
                                "SELECT pg_get_functiondef(to_regprocedure($1)) AS body",
                                signature,
                            )
                            definition = (
                                str(definition_row["body"] or "")
                                if definition_row
                                else ""
                            )
                            functions[function_name]["candidate_filters_v2"] = all(
                                marker in definition
                                for marker in SEARCH_CANDIDATE_V2_MARKERS
                            )
        except Exception as exc:
            return {
                "ready": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }

        ready = _read_contract_is_ready(vector_available, functions)
        return {
            "ready": ready,
            "vector_extension": vector_available,
            "functions": functions,
        }

    async def fetch_place_content_hashes(
        self,
        ids: Iterable[str],
        function_name: str = "get_place_content_hashes",
    ) -> dict[str, str]:
        return await self._fetch_content_hashes(
            function_name=function_name,
            ids=ids,
        )

    async def fetch_post_content_hashes(self, ids: Iterable[str]) -> dict[str, str]:
        return await self._fetch_content_hashes(
            function_name="get_post_content_hashes",
            ids=ids,
        )

    async def fetch_resource_content_hashes(
        self,
        resource_type: str,
        ids: Iterable[str],
    ) -> dict[str, str]:
        _validate_resource_type(resource_type)
        id_list = list(ids)
        if not id_list:
            return {}
        async with self.connection() as connection:
            rows = await connection.fetch(
                "SELECT * FROM get_resource_content_hashes($1::text, $2::text[])",
                resource_type,
                id_list,
            )
        return {
            str(row["external_id"]): str(row["content_hash"])
            for row in rows
            if row["content_hash"] is not None
        }

    async def upsert_place_embeddings(
        self,
        records: list[VectorUpsertRecord],
        function_name: str = "upsert_place_embedding",
        embedding_model: str | None = None,
        embedding_version: str | None = None,
    ) -> None:
        await self._upsert_records(
            function_name=function_name,
            records=records,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )

    async def upsert_post_embeddings(
        self,
        records: list[VectorUpsertRecord],
    ) -> None:
        if not records:
            return
        query = """
            SELECT upsert_post_embedding(
                $1::text, $2::text, $3::jsonb, $4::vector,
                $5::text, $6::text, $7::text, $8::boolean,
                $9::text, $10::text, $11::timestamptz, $12::bigint
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
                record.author_type,
                record.author_id,
                record.published_at,
                record.source_version,
            )
            for record in records
        ]
        async with self.connection() as connection:
            await connection.executemany(query, rows)

    async def upsert_resource_embeddings(
        self,
        resource_type: str,
        records: list[VectorUpsertRecord],
    ) -> None:
        _validate_resource_type(resource_type)
        if not records:
            return
        query = """
            SELECT upsert_resource_embedding(
                $1::text, $2::text, $3::text, $4::jsonb, $5::vector,
                $6::text, $7::text, $8::text, $9::boolean
            )
        """
        rows = [
            (
                resource_type,
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

    async def _match(
        self,
        function_name: str,
        embedding: list[float],
        filters: dict[str, Any],
        limit: int,
    ) -> list[VectorMatch]:
        query = f"SELECT * FROM {quote_identifier(function_name)}($1::vector, $2::integer, $3::jsonb)"
        try:
            async with self.connection() as connection:
                rows = await connection.fetch(
                    query,
                    vector_literal(embedding),
                    limit,
                    json.dumps(filters, ensure_ascii=False),
                )
        except asyncpg.exceptions.UndefinedFunctionError as exc:
            raise AppError(
                "Pgvector SQL contract is missing or not visible to this role. "
                "Run sql/aws_pgvector_contract.sql in the configured database and "
                "grant EXECUTE to the reader role.",
                code="pgvector_contract_missing",
                status_code=503,
            ) from exc
        return [_row_to_vector_match(row) for row in rows]

    async def _fetch_content_hashes(
        self,
        function_name: str,
        ids: Iterable[str],
    ) -> dict[str, str]:
        id_list = list(ids)
        if not id_list:
            return {}

        query = f"SELECT * FROM {quote_identifier(function_name)}($1::text[])"
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
        embedding_model: str | None = None,
        embedding_version: str | None = None,
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
                embedding_model or self._settings.embedding_model,
                embedding_version or self._settings.embedding_version,
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
        semantic_score=_optional_float(_row_value(row, "semantic_score")),
        lexical_score=_optional_float(_row_value(row, "lexical_score")),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _validate_resource_type(resource_type: str) -> None:
    if resource_type not in SEARCH_RESOURCE_TYPES:
        raise ValueError(f"Unsupported search resource type: {resource_type}")


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _build_connection_kwargs(settings: Settings, role: str) -> dict[str, Any]:
    user, password = _credentials_for_role(settings, role)
    required = {
        "PGVECTOR_HOST": settings.pgvector_host,
        "PGVECTOR_DATABASE": settings.pgvector_database,
        f"PGVECTOR_{role.upper()}_USER": user,
        f"PGVECTOR_{role.upper()}_PASSWORD": password,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing pgvector settings: {', '.join(missing)}")
    return {
        "host": settings.pgvector_host,
        "port": settings.pgvector_port,
        "database": settings.pgvector_database,
        "user": user,
        "password": password,
        "timeout": settings.request_timeout_seconds,
        "command_timeout": settings.request_timeout_seconds,
    }


def _read_contract_is_ready(
    vector_available: bool,
    functions: dict[str, dict[str, Any]],
) -> bool:
    return vector_available and all(
        details.get("exists") is True
        and details.get("executable") is True
        and (
            function_name != "search_resource_embeddings"
            or details.get("candidate_filters_v2") is True
        )
        for function_name, details in functions.items()
    )


def _configured_read_contract_signatures(
    settings: Settings,
) -> dict[str, str]:
    signatures = dict(READ_CONTRACT_SIGNATURES)
    match_name = settings.places_pgvector_match_function
    signatures[match_name] = f"{match_name}(vector, integer, jsonb)"
    hybrid_name = settings.places_pgvector_hybrid_function
    if hybrid_name:
        signatures[hybrid_name] = f"{hybrid_name}(text, vector, integer, jsonb)"
    return signatures


def _credentials_for_role(settings: Settings, role: str) -> tuple[str | None, str | None]:
    if role == "writer":
        return (
            settings.pgvector_writer_user or settings.pgvector_user,
            settings.pgvector_writer_password or settings.pgvector_password,
        )
    return (
        settings.pgvector_reader_user or settings.pgvector_user,
        settings.pgvector_reader_password or settings.pgvector_password,
    )


def _build_ssl_context(mode: str | None) -> ssl.SSLContext | None:
    if mode != "require":
        raise RuntimeError("PGVECTOR_SSL_MODE must be require")
    context = ssl.create_default_context()
    # PostgreSQL sslmode=require encrypts traffic but does not verify the CA chain.
    # This keeps compatibility with RDS certificates in slim containers without a bundled RDS CA.
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
