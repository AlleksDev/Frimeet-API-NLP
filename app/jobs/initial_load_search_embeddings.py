import asyncio

from app.jobs.sync_search_embeddings import main


if __name__ == "__main__":
    asyncio.run(main())
