import asyncio

from app.shared.config.settings import get_settings
from app.shared.logging.config import configure_logging, get_logger
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    embedding_provider = MockEmbeddingProvider()
    logger.info("Starting placeholder post embedding rebuild")
    logger.info("Target Chroma collection: %s", settings.chroma_posts_collection)
    logger.info("Embedding dimension: %s", len(embedding_provider.embed_text("sample")))
    logger.info("TODO: load posts from source DB, embed in batches and upsert to ChromaDB")


if __name__ == "__main__":
    asyncio.run(main())
