"""Open-vocabulary place category alignment over a dynamic concept catalog."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.modules.places.domain.chat_intent import PlaceCategoryInference
from app.shared.nlp.embeddings.base import EmbeddingProvider


@dataclass(frozen=True)
class PlaceCategoryConcept:
    """A category concept supplied by configuration, a database, or an API."""

    id: str
    label: str
    description: str
    examples: tuple[str, ...] = ()
    storage_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "id"))
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(
            self,
            "description",
            _required_text(self.description, "description"),
        )
        object.__setattr__(
            self,
            "examples",
            _clean_text_values(self.examples, "examples"),
        )
        object.__setattr__(
            self,
            "storage_values",
            _clean_text_values(self.storage_values, "storage_values"),
        )


@dataclass(frozen=True)
class PlaceCategoryMatch:
    """One semantic catalog match, including separation from the next match."""

    concept_id: str
    label: str
    description: str
    storage_values: tuple[str, ...]
    score: float
    margin: float


@dataclass(frozen=True)
class _IndexedConcept:
    concept: PlaceCategoryConcept
    vectors: tuple[tuple[float, ...], ...]


ConceptInput = PlaceCategoryConcept | Mapping[str, Any]


class OpenVocabularyPlaceCategoryClassifier:
    """Ranks injected concepts without a closed classifier label head.

    Concept vectors are built on first use, so constructing this component does
    not force a transformer model to load during application startup.  ``rank``
    always exposes evidence; ``classify`` applies confidence thresholds and is
    compatible with ``PlaceActivityClassifier``.
    """

    def __init__(
        self,
        concepts: Sequence[ConceptInput],
        embedding_provider: EmbeddingProvider,
        *,
        concept_embedding_provider: EmbeddingProvider | None = None,
        minimum_similarity: float = 0.44,
        minimum_margin: float = 0.04,
    ) -> None:
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1")
        if not 0.0 <= minimum_margin <= 2.0:
            raise ValueError("minimum_margin must be between 0 and 2")

        prepared_concepts = tuple(_coerce_concept(concept) for concept in concepts)
        duplicate_ids = _duplicate_concept_ids(prepared_concepts)
        if duplicate_ids:
            raise ValueError(
                "concept ids must be unique (case-insensitive): "
                + ", ".join(duplicate_ids)
            )

        self._concepts = prepared_concepts
        self._concepts_by_id = {
            concept.id.casefold(): concept for concept in prepared_concepts
        }
        self._embedding_provider = embedding_provider
        self._concept_embedding_provider = (
            concept_embedding_provider or embedding_provider
        )
        self._minimum_similarity = minimum_similarity
        self._minimum_margin = minimum_margin
        self._index: tuple[_IndexedConcept, ...] | None = None
        self._embedding_dimension: int | None = None
        self._index_lock = threading.Lock()

    @property
    def concepts(self) -> tuple[PlaceCategoryConcept, ...]:
        return self._concepts

    @property
    def is_indexed(self) -> bool:
        return self._index is not None

    def get_concept(self, concept_id: str) -> PlaceCategoryConcept | None:
        """Resolve persisted concept IDs without rerunning classification."""

        if not isinstance(concept_id, str):
            return None
        return self._concepts_by_id.get(concept_id.strip().casefold())

    def rank(self, text: str, limit: int = 3) -> tuple[PlaceCategoryMatch, ...]:
        """Return semantic top-k matches without suppressing low-confidence rows."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        query_text = text.strip()
        if not query_text or not self._concepts:
            return ()

        query_vector = _validated_vector(
            self._embedding_provider.embed_text(query_text),
            context="query",
        )
        if query_vector is None:
            return ()

        index = self._ensure_index()
        if not index:
            return ()
        if self._embedding_dimension != len(query_vector):
            raise ValueError(
                "query and concept embedding dimensions differ: "
                f"query={len(query_vector)}, concepts={self._embedding_dimension}"
            )

        scores = sorted(
            (
                (
                    entry,
                    max(
                        _cosine_similarity(query_vector, prototype)
                        for prototype in entry.vectors
                    ),
                )
                for entry in index
                if entry.vectors
            ),
            key=lambda item: (-item[1], item[0].concept.id.casefold()),
        )
        matches: list[PlaceCategoryMatch] = []
        for position, (entry, score) in enumerate(scores[:limit]):
            next_score = scores[position + 1][1] if position + 1 < len(scores) else -1.0
            concept = entry.concept
            matches.append(
                PlaceCategoryMatch(
                    concept_id=concept.id,
                    label=concept.label,
                    description=concept.description,
                    storage_values=concept.storage_values,
                    score=score,
                    margin=max(0.0, score - next_score),
                )
            )
        return tuple(matches)

    def classify(self, text: str) -> PlaceCategoryInference | None:
        """Return the best concept only when its score and margin are sufficient."""

        matches = self.rank(text, limit=2)
        if not matches:
            return None
        best = matches[0]
        if (
            best.score < self._minimum_similarity
            or best.margin < self._minimum_margin
        ):
            return None

        semantic_confidence = (best.score + 1.0) / 2.0
        separation_confidence = min(1.0, best.margin / 0.5)
        confidence = max(
            0.0,
            min(0.99, semantic_confidence * 0.85 + separation_confidence * 0.15),
        )
        return PlaceCategoryInference(
            category=best.concept_id,
            confidence=confidence,
            source="semantic_activity",
            category_values=best.storage_values,
            label=best.label,
        )

    def _ensure_index(self) -> tuple[_IndexedConcept, ...]:
        index = self._index
        if index is not None:
            return index

        with self._index_lock:
            index = self._index
            if index is not None:
                return index

            texts: list[str] = []
            owners: list[int] = []
            for concept_index, concept in enumerate(self._concepts):
                for semantic_text in _semantic_texts(concept):
                    texts.append(semantic_text)
                    owners.append(concept_index)

            raw_vectors = self._concept_embedding_provider.embed_batch(texts)
            if len(raw_vectors) != len(texts):
                raise ValueError(
                    "embedding provider returned an unexpected number of concept "
                    f"vectors: returned={len(raw_vectors)}, expected={len(texts)}"
                )

            vectors_by_concept: list[list[tuple[float, ...]]] = [
                [] for _ in self._concepts
            ]
            embedding_dimension: int | None = None
            for text_index, (owner, raw_vector) in enumerate(
                zip(owners, raw_vectors)
            ):
                vector = _validated_vector(
                    raw_vector,
                    context=f"concept text {text_index}",
                )
                if vector is None:
                    continue
                if embedding_dimension is None:
                    embedding_dimension = len(vector)
                elif len(vector) != embedding_dimension:
                    raise ValueError(
                        "concept embedding dimensions differ: "
                        f"expected={embedding_dimension}, returned={len(vector)}, "
                        f"text_index={text_index}"
                    )
                vectors_by_concept[owner].append(vector)

            index = tuple(
                _IndexedConcept(concept=concept, vectors=tuple(vectors))
                for concept, vectors in zip(self._concepts, vectors_by_concept)
                if vectors
            )
            self._embedding_dimension = embedding_dimension
            self._index = index
            return index


def _coerce_concept(value: ConceptInput) -> PlaceCategoryConcept:
    if isinstance(value, PlaceCategoryConcept):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "concept must be PlaceCategoryConcept or a mapping, got "
            f"{type(value).__name__}"
        )
    missing = [key for key in ("id", "label", "description") if key not in value]
    if missing:
        raise ValueError("concept is missing required fields: " + ", ".join(missing))
    return PlaceCategoryConcept(
        id=value["id"],
        label=value["label"],
        description=value["description"],
        examples=_value_sequence(value.get("examples", ()), "examples"),
        storage_values=_value_sequence(
            value.get("storage_values", ()), "storage_values"
        ),
    )


def _value_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a string or sequence of strings")
    return tuple(value)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _clean_text_values(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _required_text(value, f"{field_name}[{index}]")
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    return tuple(cleaned)


def _duplicate_concept_ids(
    concepts: Sequence[PlaceCategoryConcept],
) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for concept in concepts:
        key = concept.id.casefold()
        if key in seen:
            duplicates.append(concept.id)
        seen.add(key)
    return tuple(duplicates)


def _semantic_texts(concept: PlaceCategoryConcept) -> tuple[str, ...]:
    candidates = (
        f"{concept.label}. {concept.description}",
        concept.label,
        concept.description,
        *concept.examples,
    )
    return _clean_text_values(candidates, "semantic_texts")


def _validated_vector(
    raw_vector: Sequence[float],
    *,
    context: str,
) -> tuple[float, ...] | None:
    try:
        vector = tuple(float(value) for value in raw_vector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedding for {context} is not numeric") from exc
    if not vector or not any(value != 0.0 for value in vector):
        return None
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"embedding for {context} contains NaN or infinity")
    return vector


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
