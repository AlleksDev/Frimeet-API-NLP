import math
from typing import Sequence

from app.modules.posts.application.ports.post_repository import PostVectorRepository
from app.modules.posts.domain.models import PostCandidate
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


SAMPLE_POSTS = [
    {
        "id": "post_1",
        "title": "Cena tranquila en Tuxtla",
        "city": "Tuxtla Gutierrez",
        "tags": ["cena", "pareja", "restaurante"],
        "text": "Ideas para cenar tranquilo en pareja cerca del centro.",
    },
    {
        "id": "post_2",
        "title": "Atardeceres para tomar fotos",
        "city": "Tuxtla Gutierrez",
        "tags": ["fotos", "mirador", "amigos"],
        "text": "Plan corto para caminar, tomar fotos y ver el atardecer.",
    },
    {
        "id": "post_3",
        "title": "Comida regional en San Cristobal",
        "city": "San Cristobal de las Casas",
        "tags": ["comida", "familia", "regional"],
        "text": "Publicacion sobre comida chiapaneca para familias y grupos.",
    },
]


class MockPostVectorRepository(PostVectorRepository):
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._records = []
        for post in SAMPLE_POSTS:
            searchable_text = " ".join([post["title"], post["city"], post["text"]])
            self._records.append(
                {
                    "post": post,
                    "embedding": embedding_provider.embed_document(
                        prepare_for_embedding(searchable_text)
                    ),
                }
            )

    async def search(
        self,
        embedding: list[float],
        city: str | None,
        limit: int,
    ) -> Sequence[PostCandidate]:
        candidates: list[PostCandidate] = []
        for record in self._records:
            post = record["post"]
            if city and prepare_for_embedding(post["city"]) != prepare_for_embedding(city):
                continue
            score = _cosine_similarity(embedding, record["embedding"])
            candidates.append(
                PostCandidate(
                    id=post["id"],
                    title=post["title"],
                    score=score,
                    city=post["city"],
                    tags=list(post["tags"]),
                    metadata={"source": "mock"},
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
