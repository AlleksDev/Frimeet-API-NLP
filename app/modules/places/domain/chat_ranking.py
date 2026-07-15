from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContentRankingWeights:
    semantic: float = 0.45
    lexical: float = 0.30
    theme_or_reference: float = 0.25

    def __post_init__(self) -> None:
        values = (self.semantic, self.lexical, self.theme_or_reference)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("ranking weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("ranking weights must add up to one")

    def score(
        self,
        semantic_score: float,
        lexical_score: float,
        theme_or_reference_score: float,
    ) -> float:
        score = (
            self.semantic * _unit(semantic_score)
            + self.lexical * _unit(lexical_score)
            + self.theme_or_reference * _unit(theme_or_reference_score)
        )
        return _unit(score)


def normalize_bm25(score: float) -> float:
    if not math.isfinite(score) or score <= 0:
        return 0.0
    return score / (score + 1.0)


def _unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))

