from typing import Sequence

from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


class SemanticPlaceRanker(PlaceRanker):
    """Preserve the cosine-similarity order returned by PGVector."""

    engine_name = "fasttext_mean_embeddings"
    score_metric = "cosine_similarity"
    field_weights = {"document": 1}

    def __init__(self, dimension: int = 300) -> None:
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
