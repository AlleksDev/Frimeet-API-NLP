import math
from dataclasses import dataclass
from typing import Any, Sequence

from app.modules.places.infrastructure.mock_place_repository import SAMPLE_PLACES
from app.modules.posts.infrastructure.mock_post_repository import SAMPLE_POSTS
from app.modules.search.application.ports.search_provider import SearchProvider
from app.modules.search.domain.models import SearchHit, SearchResourceType
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


@dataclass(frozen=True)
class MockSearchDocument:
    id: str
    title: str
    document: str
    metadata: dict[str, Any]


class MockHybridSearchProvider(SearchProvider):
    def __init__(
        self,
        resource_type: SearchResourceType,
        embedding_provider: EmbeddingProvider,
        documents: list[MockSearchDocument],
    ) -> None:
        self.resource_type = resource_type
        self._records = [
            (document, embedding_provider.embed_text(document.document))
            for document in documents
        ]

    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        requester_id: str | None,
    ) -> Sequence[SearchHit]:
        query_terms = set(prepare_for_embedding(query).split())
        hits: list[SearchHit] = []
        for document, document_embedding in self._records:
            if not _is_visible(document.metadata, requester_id):
                continue
            document_terms = set(prepare_for_embedding(document.document).split())
            lexical_score = (
                len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
            )
            semantic_score = _cosine_similarity(embedding, document_embedding)
            exact_boost = 0.15 if prepare_for_embedding(query) in prepare_for_embedding(document.title) else 0.0
            score = min(1.0, semantic_score * 0.65 + lexical_score * 0.35 + exact_boost)
            metadata = {
                key: value
                for key, value in document.metadata.items()
                if key != "authorized_user_ids"
            }
            hits.append(
                SearchHit(
                    id=document.id,
                    resource_type=self.resource_type,
                    title=document.title,
                    subtitle=_mock_subtitle(self.resource_type, metadata),
                    score=score,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    metadata=metadata,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


def get_mock_search_documents() -> dict[SearchResourceType, list[MockSearchDocument]]:
    places = [
        MockSearchDocument(
            id=str(place["id"]),
            title=str(place["name"]),
            document=" ".join(
                str(place[key])
                for key in ("name", "category", "tags", "occasion", "short_description")
            ),
            metadata=dict(place),
        )
        for place in SAMPLE_PLACES
    ]
    posts = [
        MockSearchDocument(
            id=str(post["id"]),
            title=str(post["title"]),
            document=" ".join(
                [str(post["title"]), " ".join(post["tags"]), str(post["text"])]
            ),
            metadata=dict(post),
        )
        for post in SAMPLE_POSTS
    ]
    return {
        SearchResourceType.PLACES: places,
        SearchResourceType.POSTS: posts,
        SearchResourceType.USERS: [
            MockSearchDocument(
                id="b83ab97e-91a4-4f69-b102-b27c6092e9cb",
                title="Usuario Tester Uno",
                document="tester1 usuario tester uno biografia frimeet",
                metadata={"username": "tester1", "full_name": "Usuario Tester Uno"},
            )
        ],
        SearchResourceType.CLUBS: [
            MockSearchDocument(
                id="club_chess",
                title="Club de Ajedrez Universitario",
                document="club ajedrez universitario partidas entrenamiento semanal",
                metadata={"category": "ajedrez", "is_online": False, "is_private": False},
            )
        ],
        SearchResourceType.GROUPS: [
            MockSearchDocument(
                id="group_uni",
                title="Amigos de la Uni",
                document="amigos universidad organizar salidas",
                metadata={
                    "is_private": True,
                    "member_count": 2,
                    "authorized_user_ids": ["00000000-0000-0000-0000-000000000001"],
                },
            )
        ],
        SearchResourceType.EVENTS: [
            MockSearchDocument(
                id="event_birthday",
                title="Fiesta de Cumpleanos",
                document="fiesta cumpleanos evento independiente",
                metadata={"is_public": True, "duration_minutes": 300},
            )
        ],
    }


def _is_visible(metadata: dict[str, Any], requester_id: str | None) -> bool:
    restricted = bool(metadata.get("is_private", False)) or metadata.get("is_public") is False
    if not restricted:
        return True
    return bool(
        requester_id
        and requester_id in {str(value) for value in metadata.get("authorized_user_ids", [])}
    )


def _mock_subtitle(
    resource_type: SearchResourceType,
    metadata: dict[str, Any],
) -> str | None:
    if resource_type == SearchResourceType.USERS and metadata.get("username"):
        return f"@{metadata['username']}"
    value = metadata.get("category") or metadata.get("city") or metadata.get("text")
    return str(value)[:160] if value else None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
