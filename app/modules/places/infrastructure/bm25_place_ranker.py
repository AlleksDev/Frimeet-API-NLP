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


class Bm25PlaceRanker(PlaceRanker):
    """Rerank pgvector candidates with Okapi BM25 from Lab 3."""

    engine_name = "bm25"
    score_metric = "bm25"
    field_weights = {
        "tags": TAG_WEIGHT,
        "category": CATEGORY_WEIGHT,
        "other_text": 1,
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self._k1 = k1
        self._b = b
        self.ranking_parameters = {"k1": k1, "b": b}

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
        query_tokens = tokenize(query)
        idf_index = idf_bm25(corpus)
        average_document_length = sum(map(len, corpus)) / len(corpus)

        scored = [
            (
                bm25_score(
                    document=document,
                    query=query_tokens,
                    idf_index=idf_index,
                    average_document_length=average_document_length,
                    k1=self._k1,
                    b=self._b,
                ),
                place,
            )
            for place, document in zip(candidates, corpus)
        ]
        scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [
            replace(place, score=score)
            for score, place in scored[:limit]
        ]


def idf_bm25(corpus: Sequence[Sequence[str]]) -> dict[str, float]:
    document_count = len(corpus)
    if document_count == 0:
        return {}

    document_frequency: Counter[str] = Counter()
    for document in corpus:
        document_frequency.update(set(document))
    return {
        term: math.log(
            1 + (document_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in document_frequency.items()
    }


def bm25_score(
    document: Sequence[str],
    query: Sequence[str],
    idf_index: dict[str, float],
    average_document_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not document or average_document_length <= 0:
        return 0.0

    counts = Counter(document)
    document_length = len(document)
    score = 0.0
    for term in query:
        frequency = counts.get(term, 0)
        if frequency == 0:
            continue
        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * (
            1 - b + b * document_length / average_document_length
        )
        score += idf_index.get(term, 0.0) * numerator / denominator
    return score
