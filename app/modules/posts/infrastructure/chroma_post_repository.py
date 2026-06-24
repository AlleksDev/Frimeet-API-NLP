from typing import Any, Sequence

from app.modules.posts.application.ports.post_repository import PostVectorRepository
from app.modules.posts.domain.models import PostCandidate
from app.shared.chroma.vector_store import ChromaVectorStore


class ChromaPostVectorRepository(PostVectorRepository):
    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    async def search(
        self,
        embedding: list[float],
        city: str | None,
        limit: int,
    ) -> Sequence[PostCandidate]:
        metadata_filter = {"city": city} if city else {}
        raw_results = await self._vector_store.query(
            embedding=embedding,
            metadata_filter=metadata_filter,
            limit=limit,
        )
        return self._to_candidates(raw_results)

    def _to_candidates(self, raw_results: dict[str, Any]) -> list[PostCandidate]:
        ids = raw_results.get("ids", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        candidates: list[PostCandidate] = []
        for index, post_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            candidates.append(
                PostCandidate(
                    id=str(post_id),
                    title=str(metadata.get("title", post_id)),
                    score=score,
                    city=metadata.get("city"),
                    tags=tags,
                    metadata=dict(metadata),
                )
            )
        return candidates
