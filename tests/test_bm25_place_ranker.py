import math

import pytest

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_text import tokenize
from app.modules.places.infrastructure.bm25_place_ranker import (
    Bm25PlaceRanker,
    bm25_score,
    idf_bm25,
)


def test_idf_bm25_uses_smoothed_nonnegative_formula() -> None:
    index = idf_bm25([["cafe", "tranquil"], ["parque", "tranquil"]])

    assert index["cafe"] == pytest.approx(math.log(2.0))
    assert index["tranquil"] == pytest.approx(math.log(1.2))
    assert all(value >= 0 for value in index.values())


def test_bm25_score_is_zero_without_term_overlap() -> None:
    score = bm25_score(
        document=["cafe", "tranquil"],
        query=["muse"],
        idf_index={"cafe": 1.0, "tranquil": 1.0},
        average_document_length=2.0,
    )

    assert score == 0.0


def test_bm25_ranker_prefers_weighted_tag_match() -> None:
    places = [
        PlaceCandidate(
            id="cafe",
            name="Café",
            score=0.1,
            metadata={"tags": "cafe tranquilo"},
            document="bebidas y postres",
        ),
        PlaceCandidate(
            id="parque",
            name="Parque",
            score=0.9,
            metadata={"tags": "naturaleza ejercicio"},
            document="senderos y arboles",
        ),
    ]

    ranking = Bm25PlaceRanker().rank(
        query="un café tranquilo",
        places=places,
        filters=PlaceFilters(),
        limit=2,
    )

    assert ranking[0].id == "cafe"
    assert ranking[0].score > 0
    assert ranking[1].score == 0


def test_tokenize_normalizes_common_spanish_variants() -> None:
    assert tokenize("lugares tranquilos para cenar") == ["tranquil", "cen"]
    assert tokenize("una cena tranquila") == ["cen", "tranquil"]
