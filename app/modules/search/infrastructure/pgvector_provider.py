from typing import Sequence

from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.domain.models import SearchHit, SearchResourceType
from app.shared.vector_store.aws_pgvector import AwsPgvectorClient
from app.shared.vector_store.models import VectorMatch

_PRIVATE_METADATA_KEYS = {"authorized_user_ids", "creator_id"}


class PgvectorHybridSearchProvider(SearchProvider):
    def __init__(
        self,
        resource_type: SearchResourceType,
        vector_client: AwsPgvectorClient,
    ) -> None:
        self.resource_type = resource_type
        self._vector_client = vector_client

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        requester_id: str | None,
    ) -> Sequence[SearchHit]:
        filters = {"is_active": True}
        if requester_id:
            filters["requester_id"] = requester_id
        matches = await self._vector_client.search_resource_embeddings(
            resource_type=self.resource_type.value,
            query_text=query,
            embedding=embedding,
            filters=filters,
            limit=limit,
        )
        return [_to_search_hit(self.resource_type, match) for match in matches]


def _to_search_hit(resource_type: SearchResourceType, match: VectorMatch) -> SearchHit:
    metadata = {
        key: value
        for key, value in match.metadata.items()
        if key not in _PRIVATE_METADATA_KEYS
    }
    title = str(
        metadata.get("search_title")
        or metadata.get("name")
        or metadata.get("title")
        or metadata.get("full_name")
        or metadata.get("username")
        or match.id
    )
    subtitle = _subtitle(resource_type, metadata)
    return SearchHit(
        id=match.id,
        resource_type=resource_type,
        title=title,
        subtitle=subtitle,
        score=match.score,
        semantic_score=match.semantic_score,
        lexical_score=match.lexical_score,
        metadata=metadata,
    )


def _subtitle(resource_type: SearchResourceType, metadata: dict[str, object]) -> str | None:
    if resource_type == SearchResourceType.USERS:
        username = metadata.get("username")
        return f"@{username}" if username else None
    for key in ("category", "description", "short_description", "city"):
        value = metadata.get(key)
        if value:
            return str(value)[:160]
    return None
