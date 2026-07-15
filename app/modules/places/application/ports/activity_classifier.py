from typing import Protocol

from app.modules.places.domain.chat_intent import PlaceCategoryInference


class PlaceActivityClassifier(Protocol):
    def classify(self, text: str) -> PlaceCategoryInference | None:
        """Infer a place category from an implicit user activity."""
