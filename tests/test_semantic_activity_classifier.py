from app.modules.places.infrastructure.semantic_activity_classifier import (
    SemanticPlaceActivityClassifier,
)
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


class ControlledEmbeddingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        normalized = prepare_for_embedding(text)
        tokens = set(normalized.split())
        if "mezcla" in normalized:
            return [1.0, 1.0]
        if tokens & {"helado", "nieve", "heladeria"}:
            return [0.0, 0.0]
        if tokens & {
                "comer",
                "comida",
                "hambre",
                "tacos",
                "pizza",
                "sushi",
                "antojo",
                "platillo",
                "cocina",
                "desayunar",
                "almorzar",
                "cenar",
        }:
            return [1.0, 0.0]
        if tokens & {
                "ejercicio",
                "entrenar",
                "gimnasio",
                "deporte",
                "futbol",
                "cancha",
                "nadar",
                "fitness",
        }:
            return [0.0, 1.0]
        return [0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def test_classifier_finds_activity_inside_a_long_message() -> None:
    classifier = SemanticPlaceActivityClassifier(
        embedding_provider=ControlledEmbeddingProvider(),
    )

    result = classifier.classify(
        "tuve un dia bastante largo y ahora se me apetecen unos tacos"
    )

    assert result is not None
    assert result.category == "restaurant"
    assert result.confidence >= 0.74


def test_classifier_abstains_when_the_best_categories_are_tied() -> None:
    classifier = SemanticPlaceActivityClassifier(
        embedding_provider=ControlledEmbeddingProvider(),
    )

    result = classifier.classify("mezcla")

    assert result is None


def test_classifier_abstains_for_a_generic_single_word_request() -> None:
    classifier = SemanticPlaceActivityClassifier(
        embedding_provider=ControlledEmbeddingProvider(),
    )

    assert classifier.classify("salir") is None
