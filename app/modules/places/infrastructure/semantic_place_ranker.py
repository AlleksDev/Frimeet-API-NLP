from typing import Sequence

from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.infrastructure.place_semantic_document import (
    PLACE_SEMANTIC_FIELD_WEIGHTS,
)


class SemanticPlaceRanker(PlaceRanker):
    """Preserve the cosine-similarity order returned by PGVector."""

    engine_name = "semantic_embeddings"
    score_metric = "cosine_similarity"
    field_weights = PLACE_SEMANTIC_FIELD_WEIGHTS

    def __init__(
        self,
        dimension: int = 300,
        model_name: str = "semantic_embeddings",
    ) -> None:
        self.engine_name = model_name
        self.ranking_parameters = {"dimension": float(dimension)}

    def rank(
        self,
        query: str,
        places: Sequence[PlaceCandidate],
        filters: PlaceFilters,
        limit: int,
    ) -> list[PlaceCandidate]:
        del query, filters
        return sorted(places, key=lambda place: place.score, reverse=True)[:limit]
