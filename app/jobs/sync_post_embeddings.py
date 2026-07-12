import argparse
import asyncio

from app.modules.posts.application.use_cases.sync_post_embeddings import SyncPostEmbeddingsUseCase
from app.modules.posts.infrastructure.aws_pgvector_post_sync_repository import AwsPgvectorPostSyncRepository
from app.modules.posts.infrastructure.main_api_post_source import MainApiPostsClient
from app.shared.config.settings import get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.factory import create_embedding_provider
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient

logger = get_logger(__name__)


async def main(default_mode: str = "incremental") -> None:
    args = _parse_args(default_mode)
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.vector_store_provider != "aws_pgvector":
        raise RuntimeError("sync_post_embeddings requiere VECTOR_STORE_PROVIDER=aws_pgvector")
    use_case = SyncPostEmbeddingsUseCase(
        source=MainApiPostsClient(settings),
        repository=AwsPgvectorPostSyncRepository(
            AwsPgvectorClient(settings, role="writer")
        ),
        embedding_provider=create_embedding_provider(settings),
        embedding_model=settings.embedding_model,
        embedding_version=settings.embedding_version,
        embedding_dimension=settings.embedding_dimension,
    )
    if args.mode == "snapshot":
        result = await use_case.sync_snapshot(
            args.batch_size, args.page_limit, args.max_pages
        )
    else:
        result = await use_case.sync_incremental(
            args.batch_size, args.page_limit, args.max_pages
        )
    logger.info(
        "post sync mode=%s processed=%s skipped=%s upserted=%s "
        "deactivated=%s last_event_id=%s errors=%s",
        args.mode,
        result.processed,
        result.skipped,
        result.upserted,
        result.deactivated,
        result.last_event_id,
        len(result.errors),
    )
    if result.errors:
        raise RuntimeError("; ".join(result.errors))


def _parse_args(default_mode: str = "incremental") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sincroniza posts hacia pgvector")
    parser.add_argument(
        "--mode", choices=("snapshot", "incremental"), default=default_mode
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
