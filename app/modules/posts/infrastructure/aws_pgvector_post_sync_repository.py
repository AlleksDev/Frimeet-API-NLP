from collections.abc import Iterable

from app.modules.posts.domain.ports.post_sync_repository import PostSyncRepository
from app.modules.posts.domain.sync import PostEmbeddingRecord
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorUpsertRecord


class AwsPgvectorPostSyncRepository(PostSyncRepository):
    def __init__(self, client: AwsPgvectorClient) -> None:
        self._client = client

    async def fetch_content_hashes(self, ids: Iterable[str]) -> dict[str, str]:
        return await self._client.fetch_post_content_hashes(ids)

    async def upsert_embeddings(self, records: list[PostEmbeddingRecord]) -> None:
        await self._client.upsert_post_embeddings(
            [
                VectorUpsertRecord(
                    id=record.source.id,
                    document=record.source.document,
                    metadata=record.source.metadata,
                    embedding=record.embedding,
                    content_hash=record.versioned_hash,
                    is_active=record.source.is_active,
                    author_type=record.source.author_type,
                    author_id=record.source.author_id,
                    published_at=record.source.published_at,
                    source_version=record.source.source_version,
                )
                for record in records
            ]
        )
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT assign_posts_to_active_cluster($1::text[])",
                [record.source.id for record in records],
            )

    async def deactivate_embedding(self, post_id: str, source_version: int) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT deactivate_post_embedding($1::text, $2::bigint)",
                post_id,
                source_version,
            )

    async def get_checkpoint(self, consumer: str) -> int:
        async with self._client.connection() as connection:
            value = await connection.fetchval(
                "SELECT get_post_sync_checkpoint($1::text)", consumer
            )
        return int(value or 0)

    async def save_checkpoint(self, consumer: str, event_id: int) -> None:
        async with self._client.connection() as connection:
            await connection.execute(
                "SELECT save_post_sync_checkpoint($1::text, $2::bigint)",
                consumer,
                event_id,
            )
