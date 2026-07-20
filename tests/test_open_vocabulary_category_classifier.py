import math

import pytest

from app.modules.places.infrastructure.open_vocabulary_category_classifier import (
    OpenVocabularyPlaceCategoryClassifier,
    PlaceCategoryConcept,
)
from app.shared.nlp.embeddings.base import EmbeddingProvider


class ControlledEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.text_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        self.text_calls.append(text)
        return self.vectors.get(text, [0.0, 0.0, 0.0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [self.vectors.get(text, [0.0, 0.0, 0.0]) for text in texts]


def _catalog_vectors() -> dict[str, list[float]]:
    return {
        "Dulces horneados. Lugares con masas dulces y glaseadas": [1.0, 0.0, 0.0],
        "Dulces horneados": [1.0, 0.0, 0.0],
        "Lugares con masas dulces y glaseadas": [1.0, 0.0, 0.0],
        "quiero una dona artesanal": [1.0, 0.0, 0.0],
        "Naturaleza urbana. Espacios abiertos con vegetacion": [0.0, 1.0, 0.0],
        "Naturaleza urbana": [0.0, 1.0, 0.0],
        "Espacios abiertos con vegetacion": [0.0, 1.0, 0.0],
        "quiero caminar entre arboles": [0.0, 1.0, 0.0],
        "se me antojaron donitas": [0.9, 0.1, 0.0],
        "quiero salir": [1.0, 1.0, 0.0],
        "algo completamente distinto": [0.0, 0.0, 1.0],
    }


def _concepts() -> list[PlaceCategoryConcept]:
    return [
        PlaceCategoryConcept(
            id="sweet_baked_goods",
            label="Dulces horneados",
            description="Lugares con masas dulces y glaseadas",
            examples=("quiero una dona artesanal",),
            storage_values=("baked_goods", "pastry_vendor"),
        ),
        PlaceCategoryConcept(
            id="urban_nature",
            label="Naturaleza urbana",
            description="Espacios abiertos con vegetacion",
            examples=("quiero caminar entre arboles",),
            storage_values=("green_area",),
        ),
    ]


def test_rank_uses_injected_catalog_and_exposes_score_margin_and_storage_values() -> None:
    embeddings = ControlledEmbeddingProvider(_catalog_vectors())
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        embeddings,
    )

    assert classifier.is_indexed is False
    assert embeddings.batch_calls == []

    matches = classifier.rank("se me antojaron donitas", limit=2)

    assert classifier.is_indexed is True
    assert len(embeddings.batch_calls) == 1
    assert [match.concept_id for match in matches] == [
        "sweet_baked_goods",
        "urban_nature",
    ]
    assert matches[0].storage_values == ("baked_goods", "pastry_vendor")
    assert matches[0].score == pytest.approx(0.9938837)
    assert matches[0].margin > 0.88
    assert matches[1].margin >= 1.0
    assert classifier.get_concept("SWEET_BAKED_GOODS") == _concepts()[0]


def test_rank_limit_still_calculates_top_margin_against_runner_up() -> None:
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        ControlledEmbeddingProvider(_catalog_vectors()),
    )

    only_match = classifier.rank("se me antojaron donitas", limit=1)

    assert len(only_match) == 1
    assert only_match[0].margin > 0.88


def test_rank_supported_discards_nearest_neighbors_below_absolute_threshold() -> None:
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        ControlledEmbeddingProvider(_catalog_vectors()),
        minimum_similarity=0.5,
    )

    raw_matches = classifier.rank("algo completamente distinto", limit=2)
    supported_matches = classifier.rank_supported(
        "algo completamente distinto",
        limit=2,
    )

    assert len(raw_matches) == 2
    assert all(match.score < 0.5 for match in raw_matches)
    assert supported_matches == ()


def test_rank_supported_keeps_supported_ties_for_clarification() -> None:
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        ControlledEmbeddingProvider(_catalog_vectors()),
        minimum_similarity=0.5,
        minimum_margin=0.05,
    )

    supported_matches = classifier.rank_supported("quiero salir", limit=2)

    assert classifier.classify("quiero salir") is None
    assert [match.concept_id for match in supported_matches] == [
        "sweet_baked_goods",
        "urban_nature",
    ]
    assert supported_matches[0].score == pytest.approx(
        supported_matches[1].score
    )
    assert supported_matches[0].score > 0.5


def test_query_and_concept_encoders_can_use_distinct_e5_prefix_roles() -> None:
    vectors = _catalog_vectors()
    query_embeddings = ControlledEmbeddingProvider(
        {"se me antojaron donitas": vectors["se me antojaron donitas"]}
    )
    concept_embeddings = ControlledEmbeddingProvider(vectors)
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        query_embeddings,
        concept_embedding_provider=concept_embeddings,
    )

    matches = classifier.rank("se me antojaron donitas", limit=1)

    assert matches[0].concept_id == "sweet_baked_goods"
    assert query_embeddings.text_calls == ["se me antojaron donitas"]
    assert query_embeddings.batch_calls == []
    assert len(concept_embeddings.batch_calls) == 1


def test_classify_is_compatible_with_place_activity_classifier() -> None:
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        ControlledEmbeddingProvider(_catalog_vectors()),
        minimum_similarity=0.5,
        minimum_margin=0.05,
    )

    result = classifier.classify("se me antojaron donitas")

    assert result is not None
    assert result.category == "sweet_baked_goods"
    assert result.source == "semantic_activity"
    assert result.category_values == ("baked_goods", "pastry_vendor")
    assert result.label == "Dulces horneados"
    assert 0.0 <= result.confidence <= 0.99


def test_classify_abstains_when_candidates_are_tied_or_similarity_is_low() -> None:
    classifier = OpenVocabularyPlaceCategoryClassifier(
        _concepts(),
        ControlledEmbeddingProvider(_catalog_vectors()),
        minimum_similarity=0.5,
        minimum_margin=0.05,
    )

    assert classifier.classify("quiero salir") is None
    assert classifier.classify("algo completamente distinto") is None


def test_mapping_catalog_is_supported_without_code_level_categories() -> None:
    concept = {
        "id": "orbital_archive",
        "label": "Archivo orbital",
        "description": "Colecciones documentales sobre misiones espaciales",
        "examples": "quiero estudiar expediciones fuera de la Tierra",
        "storage_values": ["space_records"],
    }
    semantic_text = (
        "Archivo orbital. Colecciones documentales sobre misiones espaciales"
    )
    embeddings = ControlledEmbeddingProvider(
        {
            semantic_text: [1.0, 0.0, 0.0],
            "Archivo orbital": [1.0, 0.0, 0.0],
            "Colecciones documentales sobre misiones espaciales": [1.0, 0.0, 0.0],
            "quiero estudiar expediciones fuera de la Tierra": [1.0, 0.0, 0.0],
            "busco documentos de misiones espaciales": [1.0, 0.0, 0.0],
        }
    )
    classifier = OpenVocabularyPlaceCategoryClassifier([concept], embeddings)

    match = classifier.rank("busco documentos de misiones espaciales", limit=1)[0]

    assert match.concept_id == "orbital_archive"
    assert match.storage_values == ("space_records",)


def test_blank_or_zero_query_and_empty_catalog_return_no_matches() -> None:
    embeddings = ControlledEmbeddingProvider(_catalog_vectors())
    classifier = OpenVocabularyPlaceCategoryClassifier(_concepts(), embeddings)

    assert classifier.rank("   ") == ()
    assert classifier.rank("unknown") == ()
    assert classifier.classify("unknown") is None
    assert embeddings.batch_calls == []

    empty = OpenVocabularyPlaceCategoryClassifier([], embeddings)
    assert empty.rank("se me antojaron donitas") == ()


def test_catalog_and_rank_inputs_are_validated() -> None:
    embeddings = ControlledEmbeddingProvider({})
    with pytest.raises(ValueError, match="unique"):
        OpenVocabularyPlaceCategoryClassifier(
            [
                {
                    "id": "Dynamic",
                    "label": "First",
                    "description": "First concept",
                },
                {
                    "id": "dynamic",
                    "label": "Second",
                    "description": "Second concept",
                },
            ],
            embeddings,
        )
    with pytest.raises(ValueError, match="missing required fields"):
        OpenVocabularyPlaceCategoryClassifier(
            [{"id": "incomplete", "label": "Incomplete"}],
            embeddings,
        )

    classifier = OpenVocabularyPlaceCategoryClassifier([], embeddings)
    with pytest.raises(ValueError, match="positive integer"):
        classifier.rank("query", limit=0)
    with pytest.raises(TypeError, match="text must be str"):
        classifier.rank(42)  # type: ignore[arg-type]


def test_non_finite_vectors_are_rejected() -> None:
    embeddings = ControlledEmbeddingProvider(
        {"query": [math.nan, 0.0, 1.0]}
    )
    classifier = OpenVocabularyPlaceCategoryClassifier(_concepts(), embeddings)

    with pytest.raises(ValueError, match="NaN or infinity"):
        classifier.rank("query")
