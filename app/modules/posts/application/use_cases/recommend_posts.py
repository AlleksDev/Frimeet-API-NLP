from dataclasses import dataclass, field
from typing import Any

from app.modules.posts.application.ports.post_repository import PostVectorRepository
from app.modules.posts.application.ports.ranker import PostRanker
from app.modules.posts.domain.models import PostCandidate
from app.shared.cache.memory import SimpleTTLCache
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


@dataclass(frozen=True)
class RecommendPostsResult:
    query: str
    posts: list[PostCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)


class RecommendPostsUseCase:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        post_repository: PostVectorRepository,
        ranker: PostRanker,
        cache: SimpleTTLCache | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._post_repository = post_repository
        self._ranker = ranker
        self._cache = cache

    async def execute(
        self,
        query: str,
        city: str | None = None,
        limit: int = 10,
    ) -> RecommendPostsResult:
        normalized_query = prepare_for_embedding(query)
        cache_key = f"posts:{query}:{city}:{limit}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        query_embedding = self._embedding_provider.embed_query(query)
        candidates = await self._post_repository.search(
            embedding=query_embedding,
            city=city,
            limit=max(limit * 3, limit),
        )
        ranked_posts = self._ranker.rank(candidates, limit)
        result = RecommendPostsResult(
            query=query,
            posts=ranked_posts,
            metadata={
                "strategy": "semantic_search_plus_ranking",
                "used_llm": False,
                "computed_clusters_during_request": False,
            },
        )
        if self._cache:
            self._cache.set(cache_key, result)
        return result
