from typing import Sequence

from app.modules.posts.application.ports.post_repository import PostVectorRepository
from app.modules.posts.domain.models import PostCandidate
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorMatch


class AwsPgvectorPostRepository(PostVectorRepository):
    def __init__(self, vector_client: AwsPgvectorClient) -> None:
        self._vector_client = vector_client

    async def search(
        self,
        embedding: list[float],
        city: str | None,
        limit: int,
    ) -> Sequence[PostCandidate]:
        filters = {"is_active": True}
        if city:
            filters["city"] = city
        matches = await self._vector_client.match_posts(
            embedding=embedding,
            filters=filters,
            limit=limit,
        )
        return [_match_to_candidate(match) for match in matches]


def _match_to_candidate(match: VectorMatch) -> PostCandidate:
    metadata = match.metadata
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return PostCandidate(
        id=match.id,
        title=str(metadata.get("title") or metadata.get("name") or match.id),
        score=match.score,
        city=metadata.get("city"),
        tags=tags,
        metadata=metadata,
    )
