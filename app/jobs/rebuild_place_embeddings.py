import asyncio
import argparse

from app.modules.places.infrastructure.main_api_place_source import MainApiPlacesClient
from app.modules.places.infrastructure.pgvector_place_repository import PgVectorPlaceVectorRepository
from app.shared.chroma.client import ChromaHttpClientFactory
from app.shared.chroma.vector_store import ChromaVectorStore
from app.shared.config.settings import get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding
from app.shared.pgvector.client import PgVectorConnectionFactory

logger = get_logger(__name__)


async def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    embedding_provider = MockEmbeddingProvider()
    source_client = MainApiPlacesClient(settings)
    writer = NullPlaceEmbeddingWriter() if args.dry_run else build_place_embedding_writer(settings)

    logger.info("Starting place embedding rebuild from main API")
    logger.info(
        "Source API: %s%s",
        settings.main_api_base_url,
        settings.main_api_places_search_path,
    )
    logger.info("Vector store mode: %s", settings.vector_store_mode)
    logger.info("Embedding dimension: %s", len(embedding_provider.embed_text("sample")))

    if not args.dry_run:
        await writer.ensure_schema()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, object]] = []
    total = 0

    async for place in source_client.iter_places(
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    ):
        ids.append(place.id)
        documents.append(place.document)
        metadatas.append(place.metadata)

        if len(ids) >= args.batch_size:
            total += await _flush_batch(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embedding_provider=embedding_provider,
                writer=writer,
                dry_run=args.dry_run,
            )
            ids, documents, metadatas = [], [], []

    total += await _flush_batch(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embedding_provider=embedding_provider,
        writer=writer,
        dry_run=args.dry_run,
    )
    logger.info("Finished place embedding rebuild. Processed places: %s", total)


async def _flush_batch(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, object]],
    embedding_provider: MockEmbeddingProvider,
    writer: "PlaceEmbeddingWriter",
    dry_run: bool,
) -> int:
    if not ids:
        return 0

    embeddings = embedding_provider.embed_batch(
        [prepare_for_embedding(document) for document in documents]
    )

    if dry_run:
        logger.info("Dry run: prepared %s place embeddings", len(ids))
    else:
        await writer.upsert_many(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %s place embeddings", len(ids))

    return len(ids)


class PlaceEmbeddingWriter:
    async def ensure_schema(self) -> None:
        return None

    async def upsert_many(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        raise NotImplementedError


class NullPlaceEmbeddingWriter(PlaceEmbeddingWriter):
    async def upsert_many(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        return None


class ChromaPlaceEmbeddingWriter(PlaceEmbeddingWriter):
    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    async def upsert_many(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        await self._vector_store.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )


class PgVectorPlaceEmbeddingWriter(PlaceEmbeddingWriter):
    def __init__(self, repository: PgVectorPlaceVectorRepository) -> None:
        self._repository = repository

    async def ensure_schema(self) -> None:
        await self._repository.ensure_schema()

    async def upsert_many(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        await self._repository.upsert_many(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def build_place_embedding_writer(settings) -> PlaceEmbeddingWriter:
    if settings.vector_store_mode == "pgvector":
        return PgVectorPlaceEmbeddingWriter(
            PgVectorPlaceVectorRepository(
                connection_factory=PgVectorConnectionFactory(settings),
                settings=settings,
            )
        )
    if settings.vector_store_mode != "chroma":
        raise RuntimeError(
            "Set VECTOR_STORE_MODE=pgvector or VECTOR_STORE_MODE=chroma to persist embeddings."
        )
    return ChromaPlaceEmbeddingWriter(
        ChromaVectorStore(
            client_factory=ChromaHttpClientFactory(settings),
            collection_name=settings.chroma_places_collection,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild place embeddings from the main API and upsert them to ChromaDB.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and embed records without writing to ChromaDB.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
