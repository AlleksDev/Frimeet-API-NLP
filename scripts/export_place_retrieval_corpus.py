from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.modules.places.infrastructure.main_api_place_source import MainApiPlacesClient
from app.shared.config.settings import get_settings


async def _export(args: argparse.Namespace) -> None:
    settings = get_settings()
    source = MainApiPlacesClient(settings)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        async for record in source.iter_places(
            page_limit=args.page_limit,
            max_pages=args.max_pages,
        ):
            destination.write(
                json.dumps(
                    {
                        "place_id": record.id,
                        "document": record.document,
                        "metadata": record.metadata,
                        "content_hash": record.content_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    print(f"Exported {count} real place documents to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the exact place documents used by PGVector."
    )
    parser.add_argument("--output", default="artifacts/place_retrieval_corpus.jsonl")
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_export(args))


if __name__ == "__main__":
    main()
