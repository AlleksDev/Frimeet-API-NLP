import asyncio

from app.shared.config.settings import get_settings
from app.shared.logging.config import configure_logging, get_logger

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting placeholder post clustering rebuild")
    logger.info("Source Chroma collection: %s", settings.chroma_posts_collection)
    logger.info("TODO: read post embeddings, run offline clustering and persist cluster output")


if __name__ == "__main__":
    asyncio.run(main())
