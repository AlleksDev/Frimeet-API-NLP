import pytest

from app.shared.nlp.embeddings.weighted_document import build_weighted_document


def test_build_weighted_document_repeats_each_field_by_weight() -> None:
    document = build_weighted_document(
        [
            ("nombre", 1),
            ("categoria", 4),
            ("descripcion", 3),
            ("tag", 6),
        ]
    )

    assert document.split().count("nombre") == 1
    assert document.split().count("categoria") == 4
    assert document.split().count("descripcion") == 3
    assert document.split().count("tag") == 6


def test_build_weighted_document_rejects_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_weighted_document([("texto", 0)])

