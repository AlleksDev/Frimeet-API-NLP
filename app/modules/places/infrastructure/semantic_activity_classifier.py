import math
from typing import Sequence

from app.modules.places.domain.chat_intent import PlaceCategoryInference
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import tokenize_for_embeddings


_CATEGORY_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "restaurant": (
        "comer comida restaurante",
        "hambre tacos pizza sushi",
        "desayunar almorzar cenar",
        "antojo platillo cocina",
    ),
    "cafe": (
        "cafe cafeteria capuchino espresso",
        "tomar cafe merendar conversar",
        "trabajar laptop cafe",
    ),
    "park": (
        "parque picnic caminar relajarse",
        "juegos infantiles areas verdes",
        "pasear familia cesped",
    ),
    "bar": (
        "cerveza cocteles tragos bar",
        "beber copas cantina",
    ),
    "nightlife": (
        "bailar fiesta discoteca antro",
        "fiesta musica vida nocturna",
    ),
    "culture": (
        "museo arte exposicion cultura",
        "historia galeria centro cultural",
    ),
    "shopping": (
        "comprar tiendas centro comercial",
        "ropa regalos compras",
    ),
    "sports": (
        "ejercicio entrenar gimnasio deporte",
        "futbol cancha nadar fitness",
    ),
    "bakery": (
        "pan pasteles panaderia reposteria",
        "comprar pan pastel",
    ),
    "ice_cream": (
        "helado postre heladeria dulce",
        "comer helado nieve",
    ),
    "cinema": (
        "pelicula cine estreno",
        "ver pelicula pantalla",
    ),
    "library": (
        "leer estudiar libros biblioteca",
        "lectura investigacion biblioteca",
    ),
    "market": (
        "mercado tianguis productos locales",
        "puestos comprar alimentos mercado",
    ),
    "outdoors": (
        "senderismo naturaleza montana mirador",
        "aventura aire libre paisaje",
    ),
    "lodging": (
        "dormir hotel hospedaje alojamiento",
        "pasar noche hostal",
    ),
}

_GENERIC_SINGLE_TOKEN_INTENTS = {
    "divertirme",
    "pasear",
    "relajarme",
    "salir",
}


class SemanticPlaceActivityClassifier:
    """Nearest-prototype classifier over the already-loaded local embeddings."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        minimum_similarity: float = 0.44,
        minimum_margin: float = 0.04,
        window_size: int = 8,
    ) -> None:
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1")
        if not 0.0 <= minimum_margin <= 2.0:
            raise ValueError("minimum_margin must be between 0 and 2")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self._embedding_provider = embedding_provider
        self._minimum_similarity = minimum_similarity
        self._minimum_margin = minimum_margin
        self._window_size = window_size
        self._prototype_vectors = {
            category: tuple(
                vector
                for vector in embedding_provider.embed_batch(list(prototypes))
                if _has_magnitude(vector)
            )
            for category, prototypes in _CATEGORY_PROTOTYPES.items()
        }

    def classify(self, text: str) -> PlaceCategoryInference | None:
        tokens = tokenize_for_embeddings(text)
        if not tokens:
            return None
        if len(tokens) == 1 and tokens[0] in _GENERIC_SINGLE_TOKEN_INTENTS:
            return None

        query_vectors = tuple(
            vector
            for vector in self._embedding_provider.embed_batch(
                self._query_segments(tokens)
            )
            if _has_magnitude(vector)
        )
        if not query_vectors:
            return None

        scores = sorted(
            (
                (
                    category,
                    max(
                        _cosine_similarity(query, prototype)
                        for query in query_vectors
                        for prototype in prototypes
                    ),
                )
                for category, prototypes in self._prototype_vectors.items()
                if prototypes
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not scores:
            return None

        category, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else -1.0
        margin = best_score - second_score
        if (
            best_score < self._minimum_similarity
            or margin < self._minimum_margin
        ):
            return None

        confidence = min(
            0.92,
            max(
                0.74,
                0.74
                + (best_score - self._minimum_similarity) * 0.30
                + min(margin, 0.20) * 0.35,
            ),
        )
        return PlaceCategoryInference(
            category=category,
            confidence=confidence,
            source="semantic_activity",
        )

    def _query_segments(self, tokens: list[str]) -> list[str]:
        full_text = " ".join(tokens)
        if len(tokens) <= self._window_size:
            return [full_text]

        segments = [full_text]
        stride = max(2, self._window_size // 2)
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + self._window_size]
            if len(window) >= 2:
                segments.append(" ".join(window))
            if start + self._window_size >= len(tokens):
                break
        return list(dict.fromkeys(segments))


def _has_magnitude(vector: Sequence[float]) -> bool:
    return any(float(value) != 0.0 for value in vector)


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
