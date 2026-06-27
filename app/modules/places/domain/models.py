from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlaceFilters:
    city: str | None = None
    state: str | None = None
    category: str | None = None
    price_range: str | None = None
    is_active: bool | None = True
    occasion: str | None = None

    def as_metadata_filter(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "city": self.city,
                "state": self.state,
                "category": self.category,
                "price_range": self.price_range,
                "is_active": self.is_active,
                "occasion": self.occasion,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class PlaceCandidate:
    id: str
    name: str
    score: float
    category: str | None = None
    city: str | None = None
    state: str | None = None
    price_range: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None

    def to_llm_context(self) -> dict[str, Any]:
        context = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "city": self.city,
            "state": self.state,
            "price_range": self.price_range,
            "score": round(self.score, 4),
        }
        context.update(
            {
                key: value
                for key, value in self.metadata.items()
                if key in {"tags", "occasion", "short_description"}
            }
        )
        return context
