import math
from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


SAMPLE_PLACES = [
    {
        "id": "place_1",
        "name": "Cafe Selva Norte",
        "category": "cafe",
        "city": "Tuxtla Gutierrez",
        "state": "Chiapas",
        "price_range": "$$",
        "is_active": True,
        "occasion": "pareja,amigos,tranquilo",
        "tags": "cafe tranquilo postres platica",
        "short_description": "Cafe tranquilo para platicar y tomar algo ligero.",
    },
    {
        "id": "place_2",
        "name": "Mirador Los Amorosos",
        "category": "outdoors",
        "city": "Tuxtla Gutierrez",
        "state": "Chiapas",
        "price_range": "$",
        "is_active": True,
        "occasion": "pareja,familia,fotos",
        "tags": "mirador vista fotos atardecer paseo",
        "short_description": "Espacio para caminar, ver la ciudad y tomar fotos.",
    },
    {
        "id": "place_3",
        "name": "Casa del Sabor Chiapaneco",
        "category": "restaurant",
        "city": "San Cristobal de las Casas",
        "state": "Chiapas",
        "price_range": "$$",
        "is_active": True,
        "occasion": "familia,amigos,cena",
        "tags": "comida regional cena familiar restaurante",
        "short_description": "Restaurante de comida regional para grupos pequenos.",
    },
    {
        "id": "place_4",
        "name": "Patio Central",
        "category": "restaurant",
        "city": "Tuxtla Gutierrez",
        "state": "Chiapas",
        "price_range": "$$$",
        "is_active": True,
        "occasion": "pareja,cena,tranquilo",
        "tags": "cena tranquila restaurante pareja terraza",
        "short_description": "Restaurante para una cena relajada.",
    },
]


class MockPlaceVectorRepository(PlaceVectorRepository):
    """In-memory repository used for local development and tests."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._records = []
        for place in SAMPLE_PLACES:
            searchable_text = " ".join(
                [
                    place["name"],
                    place["category"],
                    place["city"],
                    place["state"],
                    place["tags"],
                    place["short_description"],
                ]
            )
            self._records.append(
                {
                    "place": place,
                    "embedding": embedding_provider.embed_text(
                        prepare_for_embedding(searchable_text)
                    ),
                }
            )

    async def search(
        self,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        candidates: list[PlaceCandidate] = []
        for record in self._records:
            place = record["place"]
            if not self._matches_filters(place, filters):
                continue
            score = _cosine_similarity(embedding, record["embedding"])
            candidates.append(
                PlaceCandidate(
                    id=place["id"],
                    name=place["name"],
                    score=score,
                    category=place["category"],
                    city=place["city"],
                    state=place["state"],
                    price_range=place["price_range"],
                    metadata={
                        "is_active": place["is_active"],
                        "occasion": place["occasion"],
                        "tags": place["tags"],
                        "short_description": place["short_description"],
                    },
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _matches_filters(place: dict[str, object], filters: PlaceFilters) -> bool:
        metadata_filters = filters.as_metadata_filter()
        for key, expected in metadata_filters.items():
            actual = place.get(key)
            if key == "occasion":
                if _normalize(str(expected)) not in _normalize(str(actual)):
                    return False
                continue
            if _normalize(str(actual)) != _normalize(str(expected)):
                return False
        return True


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _normalize(value: str) -> str:
    return prepare_for_embedding(value)
