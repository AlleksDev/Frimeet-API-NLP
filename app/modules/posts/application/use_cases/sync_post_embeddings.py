from app.modules.posts.domain.ports.embedding_provider import EmbeddingProvider
from app.modules.posts.domain.ports.post_sync_repository import PostSyncRepository
from app.modules.posts.domain.ports.post_sync_source import PostSyncSource
from app.modules.posts.domain.sync import PostEmbeddingRecord, PostSourceRecord, PostSyncResult


class SyncPostEmbeddingsUseCase:
    CONSUMER = "post-embeddings-v1"

    def __init__(
        self,
        source: PostSyncSource,
        repository: PostSyncRepository,
        embedding_provider: EmbeddingProvider,
        embedding_model: str,
        embedding_version: str,
        embedding_dimension: int,
    ) -> None:
        self._source = source
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._model = embedding_model
        self._version = embedding_version
        self._dimension = embedding_dimension

    async def sync_snapshot(
        self, batch_size: int = 100, page_limit: int | None = None, max_pages: int | None = None
    ) -> PostSyncResult:
        result = PostSyncResult()
        batch: list[PostSourceRecord] = []
        async for post in self._source.iter_snapshot(page_limit, max_pages):
            batch.append(post)
            if len(batch) >= batch_size:
                await self._upsert_batch(batch, result)
                batch = []
        await self._upsert_batch(batch, result)
        return result

    async def sync_incremental(
        self, batch_size: int = 100, page_limit: int | None = None, max_pages: int | None = None
    ) -> PostSyncResult:
        result = PostSyncResult()
        after_id = await self._repository.get_checkpoint(self.CONSUMER)
        pending: list[PostSourceRecord] = []
        pending_last_event_id = after_id
        async for change in self._source.iter_changes(after_id, page_limit, max_pages):
            result.processed += 1
            try:
                if change.operation in {"delete", "archive", "deactivate"}:
                    if pending:
                        await self._upsert_batch(pending, result, count_processed=False)
                        await self._repository.save_checkpoint(
                            self.CONSUMER, pending_last_event_id
                        )
                        result.last_event_id = pending_last_event_id
                        pending = []
                    await self._repository.deactivate_embedding(
                        change.post_id, change.source_version
                    )
                    result.deactivated += 1
                    await self._repository.save_checkpoint(
                        self.CONSUMER, change.event_id
                    )
                    result.last_event_id = change.event_id
                elif change.post is not None:
                    pending.append(change.post)
                    pending_last_event_id = change.event_id
                    if len(pending) >= batch_size:
                        await self._upsert_batch(pending, result, count_processed=False)
                        await self._repository.save_checkpoint(
                            self.CONSUMER, pending_last_event_id
                        )
                        result.last_event_id = pending_last_event_id
                        pending = []
                else:
                    raise ValueError("un cambio upsert requiere el post completo")
            except Exception as exc:
                result.errors.append(f"event_id={change.event_id}: {exc}")
                break
        if pending and not result.errors:
            await self._upsert_batch(pending, result, count_processed=False)
            await self._repository.save_checkpoint(
                self.CONSUMER, pending_last_event_id
            )
            result.last_event_id = pending_last_event_id
        return result

    async def _upsert_batch(
        self,
        batch: list[PostSourceRecord],
        result: PostSyncResult,
        *,
        count_processed: bool = True,
    ) -> None:
        if not batch:
            return
        if count_processed:
            result.processed += len(batch)
        hashes = await self._repository.fetch_content_hashes(item.id for item in batch)
        missing_versions = [item.id for item in batch if item.source_version is None]
        if missing_versions:
            raise ValueError(
                "source_version es obligatorio para posts: "
                + ",".join(missing_versions[:20])
            )
        expected = {
            item.id: self._versioned_hash(item.content_hash, item.source_version)
            for item in batch
        }
        changed = [item for item in batch if hashes.get(item.id) != expected[item.id]]
        result.skipped += len(batch) - len(changed)
        if not changed:
            return
        vectors = self._embedding_provider.embed_batch([item.document for item in changed])
        if len(vectors) != len(changed):
            raise ValueError("el proveedor de embeddings devolvio un lote incompleto")
        records = [
            PostEmbeddingRecord(item, vector, expected[item.id])
            for item, vector in zip(changed, vectors)
        ]
        await self._repository.upsert_embeddings(records)
        result.upserted += len(records)

    def _versioned_hash(self, source_hash: str, source_version: int | None) -> str:
        from hashlib import sha256

        raw = (
            f"{source_hash}:{self._model}:{self._version}:"
            f"{self._dimension}:{source_version}"
        )
        return sha256(raw.encode("utf-8")).hexdigest()
