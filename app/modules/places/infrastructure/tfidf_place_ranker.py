from collections import Counter
from dataclasses import replace
import math
import re
from typing import Any, Sequence

from app.modules.places.application.ports.ranker import PlaceRanker
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.shared.nlp.preprocessing.text import prepare_for_embedding


TAG_WEIGHT = 6
CATEGORY_WEIGHT = 2
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class TfidfPlaceRanker(PlaceRanker):
    """Rerank retrieved candidates with the TF-IDF flow from Lab 2."""

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

        corpus = [_place_tokens(place) for place in candidates]
        idf_index = idf(corpus)
        query_vector = tfidf(_tokenize(query), idf_index)

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


def _place_tokens(place: PlaceCandidate) -> list[str]:
    tags = _as_text(place.metadata.get("tags"))
    category = place.category or ""
    base_document = place.document or " ".join(
        value
        for value in [
            place.name,
            category,
            place.city or "",
            place.state or "",
            tags,
            _as_text(place.metadata.get("occasion")),
            _as_text(place.metadata.get("short_description")),
        ]
        if value
    )

    weighted_fields = [base_document]
    weighted_fields.extend([category] * (CATEGORY_WEIGHT - 1))
    weighted_fields.extend([tags] * (TAG_WEIGHT - 1))
    return _tokenize(" ".join(value for value in weighted_fields if value))


def _tokenize(text: str) -> list[str]:
    normalized = prepare_for_embedding(text)
    return TOKEN_PATTERN.findall(normalized)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)
