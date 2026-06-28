import argparse
import asyncio
from dataclasses import dataclass

from app.modules.posts.infrastructure.main_api_post_source import (
    MainApiPostsClient,
    PostSourceRecord,
)
from app.shared.config.settings import Settings, get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.factory import create_embedding_provider
from app.shared.nlp.embeddings.versioning import versioned_embedding_hash
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorUpsertRecord

logger = get_logger(__name__)


@dataclass
class SyncCounters:
    processed: int = 0
    skipped: int = 0
    upserted: int = 0
    errors: int = 0


async def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("sync_post_embeddings requires VECTOR_STORE_PROVIDER=aws_pgvector")

    source = MainApiPostsClient(settings)
    vector_client = AwsPgvectorClient(settings, role="writer")
    embedding_provider = create_embedding_provider(settings)
    counters = SyncCounters()
    batch: list[PostSourceRecord] = []

    logger.info("Starting post embedding sync")
    logger.info("Embedding model=%s version=%s", settings.embedding_model, settings.embedding_version)

    async for post in source.iter_posts(
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    ):
        batch.append(post)
        if len(batch) >= args.batch_size:
            await _flush_batch(
                batch,
                vector_client,
                embedding_provider,
                settings,
                counters,
                args.dry_run,
            )
            batch = []

    await _flush_batch(
        batch,
        vector_client,
        embedding_provider,
        settings,
        counters,
        args.dry_run,
    )
    logger.info(
        "Finished post sync processed=%s skipped=%s upserted=%s errors=%s",
        counters.processed,
        counters.skipped,
        counters.upserted,
        counters.errors,
    )


async def _flush_batch(
    batch: list[PostSourceRecord],
    vector_client: AwsPgvectorClient,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    counters: SyncCounters,
    dry_run: bool,
) -> None:
    if not batch:
        return

    counters.processed += len(batch)
    try:
        existing_hashes = await vector_client.fetch_post_content_hashes(
            [record.id for record in batch]
        )
        expected_hashes = {
            record.id: versioned_embedding_hash(
                source_content_hash=record.content_hash,
                model=settings.embedding_model,
                version=settings.embedding_version,
                dimension=settings.embedding_dimension,
            )
            for record in batch
        }
        changed = [
            record
            for record in batch
            if existing_hashes.get(record.id) != expected_hashes[record.id]
        ]
        counters.skipped += len(batch) - len(changed)
        if not changed:
            return

        embeddings = embedding_provider.embed_batch(
            [record.document for record in changed]
        )
        upserts = [
            VectorUpsertRecord(
                id=record.id,
                document=record.document,
                metadata=record.metadata,
                embedding=embedding,
                content_hash=expected_hashes[record.id],
                is_active=record.is_active,
            )
            for record, embedding in zip(changed, embeddings)
        ]
        if dry_run:
            logger.info("Dry run: prepared %s post upserts", len(upserts))
        else:
            await vector_client.upsert_post_embeddings(upserts)
        counters.upserted += len(upserts)
    except Exception:
        counters.errors += len(batch)
        logger.exception("Failed to sync post embedding batch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync post embeddings into AWS RDS pgvector.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
