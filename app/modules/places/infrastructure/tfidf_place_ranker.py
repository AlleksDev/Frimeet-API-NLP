from collections import Counter
from dataclasses import replace
import math
from typing import Sequence

from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_text import (
    CATEGORY_WEIGHT,
    TAG_WEIGHT,
    place_tokens,
    tokenize,
)


class TfidfPlaceRanker(PlaceRanker):
    """Rerank retrieved candidates with the TF-IDF flow from Lab 2."""

    engine_name = "tfidf"
    score_metric = "cosine_similarity"
    field_weights = {
        "tags": TAG_WEIGHT,
        "category": CATEGORY_WEIGHT,
        "other_text": 1,
    }
    ranking_parameters: dict[str, float] = {}

    def rank(
        self,
        query: str,
        places: Sequence[PlaceCandidate],
        filters: PlaceFilters,
        limit: int,
    ) -> list[PlaceCandidate]:
        del filters
        candidates = list(places)
        if not candidates:
            return []

        corpus = [place_tokens(place) for place in candidates]
        idf_index = idf(corpus)
        query_vector = tfidf(tokenize(query), idf_index)

        scored = [
            (cosine_similarity(query_vector, tfidf(document, idf_index)), place)
            for place, document in zip(candidates, corpus)
        ]
        scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)

        return [
            replace(place, score=score)
            for score, place in scored[:limit]
        ]


def tf(document: Sequence[str]) -> dict[str, float]:
    total = len(document)
    if total == 0:
        return {}
    counts = Counter(document)
    return {term: count / total for term, count in counts.items()}


def idf(corpus: Sequence[Sequence[str]]) -> dict[str, float]:
    document_count = len(corpus)
    if document_count == 0:
        return {}

    document_frequency: Counter[str] = Counter()
    for document in corpus:
        document_frequency.update(set(document))
    return {
        term: math.log(document_count / frequency)
        for term, frequency in document_frequency.items()
    }


def tfidf(document: Sequence[str], idf_index: dict[str, float]) -> dict[str, float]:
    return {
        term: frequency * idf_index.get(term, 0.0)
        for term, frequency in tf(document).items()
    }


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    dot_product = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight**2 for weight in left.values()))
    right_norm = math.sqrt(sum(weight**2 for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
