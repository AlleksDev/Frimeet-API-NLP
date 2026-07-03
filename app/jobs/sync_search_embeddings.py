import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from app.modules.clubs.infrastructure.main_api_club_source import MainApiClubsClient
from app.modules.events.infrastructure.main_api_event_source import MainApiEventsClient
from app.modules.groups.infrastructure.main_api_group_source import MainApiGroupsClient
from app.modules.users.infrastructure.main_api_user_source import MainApiUsersClient
from app.shared.config.settings import Settings, get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.factory import create_embedding_provider
from app.shared.nlp.embeddings.versioning import versioned_embedding_hash
from app.shared.search.source import PagedMainApiSearchClient, SearchSourceRecord
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorUpsertRecord

logger = get_logger(__name__)

SOURCE_FACTORIES: dict[str, Callable[[Settings], PagedMainApiSearchClient]] = {
    "users": MainApiUsersClient,
    "clubs": MainApiClubsClient,
    "groups": MainApiGroupsClient,
    "events": MainApiEventsClient,
}


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
        raise RuntimeError("sync_search_embeddings requires VECTOR_STORE_PROVIDER=aws_pgvector")

    resources = list(SOURCE_FACTORIES) if args.resource == "all" else [args.resource]
    embedding_provider = create_embedding_provider(settings)
    vector_client = AwsPgvectorClient(settings, role="writer")
    total_errors = 0
    for resource_type in resources:
        counters = await sync_resource(
            resource_type=resource_type,
            source=SOURCE_FACTORIES[resource_type](settings),
            vector_client=vector_client,
            embedding_provider=embedding_provider,
            settings=settings,
            batch_size=args.batch_size,
            page_limit=args.page_limit,
            max_pages=args.max_pages,
            dry_run=args.dry_run,
        )
        total_errors += counters.errors
    if total_errors:
        raise RuntimeError(f"Search embedding sync finished with {total_errors} failed records")


async def sync_resource(
    *,
    resource_type: str,
    source: PagedMainApiSearchClient,
    vector_client: AwsPgvectorClient,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    batch_size: int = 100,
    page_limit: int | None = None,
    max_pages: int | None = None,
    dry_run: bool = False,
) -> SyncCounters:
    counters = SyncCounters()
    batch: list[SearchSourceRecord] = []
    logger.info("Starting %s embedding sync", resource_type)
    async for record in source.iter_records(page_limit=page_limit, max_pages=max_pages):
        batch.append(record)
        if len(batch) >= batch_size:
            await _flush_batch(
                resource_type, batch, vector_client, embedding_provider, settings, counters, dry_run
            )
            batch = []
    await _flush_batch(
        resource_type, batch, vector_client, embedding_provider, settings, counters, dry_run
    )
    logger.info(
        "Finished %s sync processed=%s skipped=%s upserted=%s errors=%s",
        resource_type,
        counters.processed,
        counters.skipped,
        counters.upserted,
        counters.errors,
    )
    return counters


async def _flush_batch(
    resource_type: str,
    batch: list[SearchSourceRecord],
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
        existing_hashes = await vector_client.fetch_resource_content_hashes(
            resource_type, [record.id for record in batch]
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
        embeddings = embedding_provider.embed_batch([record.document for record in changed])
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
        if not dry_run:
            await vector_client.upsert_resource_embeddings(resource_type, upserts)
        counters.upserted += len(upserts)
    except Exception:
        counters.errors += len(batch)
        logger.exception("Failed to sync %s embedding batch", resource_type)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync users, clubs, groups and events into AWS RDS pgvector."
    )
    parser.add_argument("--resource", choices=[*SOURCE_FACTORIES, "all"], required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
