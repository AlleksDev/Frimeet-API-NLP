import argparse
import asyncio
from dataclasses import dataclass

from app.modules.places.infrastructure.main_api_place_source import (
    MainApiPlacesClient,
    PlaceSourceRecord,
)
from app.shared.config.settings import Settings, get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.factory import create_place_embedding_provider
from app.shared.nlp.embeddings.versioning import versioned_embedding_hash
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorUpsertRecord

logger = get_logger(__name__)


@dataclass
class SyncCounters:
    processed: int = 0
    skipped: int = 0
    upserted: int = 0
    deactivated: int = 0
    errors: int = 0


async def main(default_mode: str = "incremental") -> None:
    args = _parse_args(default_mode)
    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("sync_place_embeddings requires VECTOR_STORE_PROVIDER=aws_pgvector")
    _validate_backfill_configuration(settings)

    source = MainApiPlacesClient(settings)
    vector_client = AwsPgvectorClient(settings, role="writer")
    embedding_provider = create_place_embedding_provider(settings, text_role="passage")
    counters = SyncCounters()
    batch: list[PlaceSourceRecord] = []

    logger.info("Starting place embedding sync")
    logger.info(
        "Places embedding model=%s revision=%s version=%s dimension=%s "
        "fix_mistral_regex=%s",
        settings.places_embedding_model,
        settings.places_embedding_model_revision or "unpinned",
        settings.places_embedding_version,
        settings.places_embedding_dimension,
        settings.places_embedding_fix_mistral_regex,
    )

    if args.mode == "incremental":
        await _sync_incremental(
            source,
            vector_client,
            embedding_provider,
            settings,
            counters,
            args,
        )
    else:
        async for place in source.iter_places(
            page_limit=args.page_limit,
            max_pages=args.max_pages,
        ):
            batch.append(place)
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
        "Finished place sync mode=%s processed=%s skipped=%s upserted=%s "
        "deactivated=%s errors=%s",
        args.mode,
        counters.processed,
        counters.skipped,
        counters.upserted,
        counters.deactivated,
        counters.errors,
    )
    if counters.errors:
        raise RuntimeError(
            f"Place embedding sync finished with {counters.errors} failed records"
        )


async def _sync_incremental(
    source: MainApiPlacesClient,
    vector_client: AwsPgvectorClient,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    counters: SyncCounters,
    args: argparse.Namespace,
) -> None:
    consumer = f"place-embeddings:{settings.places_pgvector_upsert_function}"
    deactivate_function = (
        "deactivate_place_embedding_semantic_v1"
        if settings.places_pgvector_upsert_function
        == "upsert_place_embedding_semantic_v1"
        else "deactivate_place_embedding"
    )
    checkpoint = await vector_client.get_place_sync_checkpoint(consumer)
    batch: list[PlaceSourceRecord] = []
    batch_last_event_id = checkpoint

    async for change in source.iter_changes(
        after_event_id=checkpoint,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    ):
        if change.operation == "deactivate":
            if batch:
                if not await _flush_batch(
                    batch,
                    vector_client,
                    embedding_provider,
                    settings,
                    counters,
                    args.dry_run,
                ):
                    return
                if not args.dry_run:
                    await vector_client.save_place_sync_checkpoint(
                        consumer, batch_last_event_id
                    )
                batch = []
            counters.processed += 1
            if not args.dry_run:
                await vector_client.deactivate_place_embedding(
                    change.place_id,
                    function_name=deactivate_function,
                )
                await vector_client.save_place_sync_checkpoint(
                    consumer, change.event_id
                )
            counters.deactivated += 1
            continue

        if change.place is None:
            counters.errors += 1
            logger.error(
                "Place change event_id=%s is missing its source record",
                change.event_id,
            )
            return
        batch.append(change.place)
        batch_last_event_id = change.event_id
        if len(batch) >= args.batch_size:
            if not await _flush_batch(
                batch,
                vector_client,
                embedding_provider,
                settings,
                counters,
                args.dry_run,
            ):
                return
            if not args.dry_run:
                await vector_client.save_place_sync_checkpoint(
                    consumer, batch_last_event_id
                )
            batch = []

    if batch and await _flush_batch(
        batch,
        vector_client,
        embedding_provider,
        settings,
        counters,
        args.dry_run,
    ):
        if not args.dry_run:
            await vector_client.save_place_sync_checkpoint(
                consumer, batch_last_event_id
            )


async def _flush_batch(
    batch: list[PlaceSourceRecord],
    vector_client: AwsPgvectorClient,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    counters: SyncCounters,
    dry_run: bool,
) -> bool:
    if not batch:
        return True

    counters.processed += len(batch)
    try:
        existing_hashes = await vector_client.fetch_place_content_hashes(
            [record.id for record in batch],
            function_name=settings.places_pgvector_hash_function,
        )
        expected_hashes = {
            record.id: versioned_embedding_hash(
                source_content_hash=record.content_hash,
                model=settings.places_embedding_model,
                version=settings.places_embedding_version,
                dimension=settings.places_embedding_dimension,
                revision=settings.places_embedding_model_revision,
                fix_mistral_regex=(
                    settings.places_embedding_fix_mistral_regex
                    if settings.places_embedding_provider
                    in {"sentence_transformer", "bert"}
                    else None
                ),
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
            return True

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
            logger.info("Dry run: prepared %s place upserts", len(upserts))
        else:
            await vector_client.upsert_place_embeddings(
                upserts,
                function_name=settings.places_pgvector_upsert_function,
                embedding_model=settings.places_embedding_model,
                embedding_version=settings.places_embedding_version,
            )
            counters.upserted += len(upserts)
        return True
    except Exception:
        counters.errors += len(batch)
        logger.exception("Failed to sync place embedding batch")
        return False


def _validate_backfill_configuration(settings: Settings) -> None:
    """Fail early when the 768d model and database writers are mixed."""

    semantic_upsert = "upsert_place_embedding_semantic_v1"
    semantic_hash = "get_place_content_hashes_semantic_v1"
    selected_upsert = settings.places_pgvector_upsert_function
    selected_hash = settings.places_pgvector_hash_function
    uses_semantic_upsert = selected_upsert == semantic_upsert
    uses_semantic_hash = selected_hash == semantic_hash

    if uses_semantic_upsert != uses_semantic_hash:
        raise RuntimeError(
            "The semantic Places backfill requires both writer functions: "
            f"PLACES_PGVECTOR_UPSERT_FUNCTION={semantic_upsert} and "
            f"PLACES_PGVECTOR_HASH_FUNCTION={semantic_hash}"
        )

    semantic_provider = settings.places_embedding_provider in {
        "sentence_transformer",
        "bert",
    }
    if uses_semantic_upsert:
        if not semantic_provider or settings.places_embedding_dimension != 768:
            raise RuntimeError(
                "place_embeddings_semantic_v1 requires a Sentence-Transformer "
                "provider with PLACES_EMBEDDING_DIMENSION=768"
            )
        if settings.places_embedding_model_revision is None:
            logger.warning(
                "PLACES_EMBEDDING_MODEL_REVISION is not pinned; use a full "
                "Hugging Face commit SHA for reproducible production vectors"
            )
        if not settings.places_embedding_passage_prefix:
            logger.warning(
                "PLACES_EMBEDDING_PASSAGE_PREFIX is empty; E5-family models "
                "normally require the 'passage:' prefix"
            )
    elif semantic_provider and settings.places_embedding_dimension == 768:
        raise RuntimeError(
            "A 768d Places retriever cannot write through the legacy 300d "
            "functions. Configure PLACES_PGVECTOR_UPSERT_FUNCTION="
            f"{semantic_upsert} and PLACES_PGVECTOR_HASH_FUNCTION={semantic_hash}"
        )


def _parse_args(default_mode: str = "incremental") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync place embeddings into AWS RDS pgvector.")
    parser.add_argument(
        "--mode",
        choices=("snapshot", "incremental"),
        default=default_mode,
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
