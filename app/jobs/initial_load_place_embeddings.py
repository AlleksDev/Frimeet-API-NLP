import asyncio

from app.jobs.sync_place_embeddings import main


if __name__ == "__main__":
    asyncio.run(main())
