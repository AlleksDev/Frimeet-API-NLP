import math
from dataclasses import dataclass

from app.modules.search.domain.models import SearchHit, SearchResourceType


@dataclass(frozen=True)
class SearchRelevanceThreshold:
    semantic_min: float
    lexical_min: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.semantic_min) or not -1.0 <= self.semantic_min <= 1.0:
            raise ValueError("semantic_min must be finite and between -1 and 1")
        if not math.isfinite(self.lexical_min) or self.lexical_min < 0.0:
            raise ValueError("lexical_min must be finite and non-negative")

    def accepts(self, hit: SearchHit) -> bool:
        semantic_match = (
            hit.semantic_score is not None
            and math.isfinite(hit.semantic_score)
            and hit.semantic_score >= self.semantic_min
        )
        lexical_match = (
            hit.lexical_score is not None
            and math.isfinite(hit.lexical_score)
            and hit.lexical_score >= self.lexical_min
        )
        return semantic_match or lexical_match


@dataclass(frozen=True)
class SearchRelevancePolicy:
    thresholds: dict[SearchResourceType, SearchRelevanceThreshold]
    version: str

    def __post_init__(self) -> None:
        missing = set(SearchResourceType) - set(self.thresholds)
        extra = set(self.thresholds) - set(SearchResourceType)
        if missing or extra:
            raise ValueError("relevance policy must define every search resource")
        if not self.version.strip() or len(self.version) > 64:
            raise ValueError("relevance policy version is invalid")

    def threshold_for(
        self, resource_type: SearchResourceType
    ) -> SearchRelevanceThreshold:
        return self.thresholds[resource_type]

    def accepts(self, hit: SearchHit) -> bool:
        return self.threshold_for(hit.resource_type).accepts(hit)

    @classmethod
    def uniform(
        cls,
        semantic_min: float = 0.30,
        lexical_min: float = 0.05,
        version: str = "global-search-relevance-v1",
    ) -> "SearchRelevancePolicy":
        return cls(
            thresholds={
                resource_type: SearchRelevanceThreshold(
                    semantic_min=semantic_min,
                    lexical_min=lexical_min,
                )
                for resource_type in SearchResourceType
            },
            version=version,
        )
