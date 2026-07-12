from datetime import UTC, datetime

import pytest

from app.modules.posts.application.use_cases.sync_post_embeddings import SyncPostEmbeddingsUseCase
from app.modules.posts.domain.sync import PostChangeRecord, PostSourceRecord


class SourceStub:
    def __init__(self, changes):
        self.changes = changes

    async def iter_snapshot(self, page_limit=None, max_pages=None):
        if False:
            yield None

    async def iter_changes(self, after_id, page_limit=None, max_pages=None):
        for item in self.changes:
            yield item


class RepositoryStub:
    def __init__(self):
        self.checkpoint = 0
        self.upserted = []
        self.deactivated = []

    async def fetch_content_hashes(self, ids):
        return {}

    async def upsert_embeddings(self, records):
        self.upserted.extend(records)

    async def deactivate_embedding(self, post_id, source_version):
        self.deactivated.append((post_id, source_version))

    async def get_checkpoint(self, consumer):
        return self.checkpoint

    async def save_checkpoint(self, consumer, event_id):
        self.checkpoint = event_id


class EmbeddingStub:
    def embed_text(self, text):
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_incremental_sync_upserts_and_deactivates_in_order() -> None:
    post = PostSourceRecord(
        "p1", "cafe", {}, "hash", True, "user", "u1", datetime.now(UTC), 1
    )
    source = SourceStub(
        [
            PostChangeRecord(1, "p1", "upsert", 1, post),
            PostChangeRecord(2, "p2", "delete", 2, None),
        ]
    )
    repository = RepositoryStub()
    use_case = SyncPostEmbeddingsUseCase(
        source, repository, EmbeddingStub(), "model", "v1", 2
    )
    result = await use_case.sync_incremental(batch_size=10)
    assert [item.source.id for item in repository.upserted] == ["p1"]
    assert repository.deactivated == [("p2", 2)]
    assert repository.checkpoint == 2
    assert result.errors == []
