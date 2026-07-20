"""Backward-compatible facade for the data-driven category aligner.

The previous implementation embedded a fixed category/prototype dictionary in
Python.  This facade keeps the import path for callers while requiring concepts
to come from configuration, the source API, or a catalog export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.modules.places.domain.chat_intent import PlaceCategoryInference
from app.modules.places.infrastructure.open_vocabulary_category_classifier import (
    OpenVocabularyPlaceCategoryClassifier,
    PlaceCategoryConcept,
    PlaceCategoryMatch,
)
from app.shared.nlp.embeddings.base import EmbeddingProvider


class SemanticPlaceActivityClassifier:
    """Compatibility adapter with no built-in categories or lexical rules."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        concepts: Sequence[PlaceCategoryConcept | Mapping[str, Any]],
        *,
        concept_embedding_provider: EmbeddingProvider | None = None,
        minimum_similarity: float = 0.44,
        minimum_margin: float = 0.04,
        window_size: int | None = None,
    ) -> None:
        if window_size is not None and window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._delegate = OpenVocabularyPlaceCategoryClassifier(
            concepts=concepts,
            embedding_provider=embedding_provider,
            concept_embedding_provider=concept_embedding_provider,
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )

    @property
    def concepts(self) -> tuple[PlaceCategoryConcept, ...]:
        return self._delegate.concepts

    def get_concept(self, concept_id: str) -> PlaceCategoryConcept | None:
        return self._delegate.get_concept(concept_id)

    def classify(self, text: str) -> PlaceCategoryInference | None:
        return self._delegate.classify(text)

    def rank(self, text: str, limit: int = 3) -> tuple[PlaceCategoryMatch, ...]:
        return self._delegate.rank(text, limit=limit)

    def rank_supported(
        self,
        text: str,
        limit: int = 3,
    ) -> tuple[PlaceCategoryMatch, ...]:
        return self._delegate.rank_supported(text, limit=limit)
