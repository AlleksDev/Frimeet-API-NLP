from collections.abc import Iterable

from app.shared.nlp.preprocessing.text import clean_text


def build_weighted_document(
    fields: Iterable[tuple[str | None, int]],
) -> str:
    """Approximate field weights by repeating text before mean pooling."""
    weighted_parts: list[str] = []
    for value, weight in fields:
        if weight < 1:
            raise ValueError("Semantic field weights must be positive integers")
        normalized = clean_text(value or "")
        if normalized:
            weighted_parts.extend([normalized] * weight)
    return clean_text(" ".join(weighted_parts))

