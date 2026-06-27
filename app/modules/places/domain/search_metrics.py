from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class SearchMetricValues:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    map: float
    ndcg_at_k: float


@dataclass(frozen=True)
class SearchEngineMetrics:
    engine: str
    candidate_retrieval: str
    score_metric: str
    field_weights: dict[str, int]
    candidate_count: int
    returned_count: int
    nonzero_score_count: int
    min_score: float
    max_score: float
    mean_score: float
    location_filter_applied: bool
    nearby_place_count: int | None
    radius_meters: int | None


def evaluate_ranking(
    ranking: Sequence[str],
    relevance: dict[str, int],
    k: int,
) -> SearchMetricValues:
    top_k = list(ranking[:k])
    relevant_ids = {
        place_id for place_id, grade in relevance.items() if grade > 0
    }
    retrieved_relevant = [
        place_id for place_id in top_k if place_id in relevant_ids
    ]

    precision = len(retrieved_relevant) / k
    recall = (
        len(set(retrieved_relevant)) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0
    for position, place_id in enumerate(top_k, start=1):
        if place_id in relevant_ids:
            reciprocal_rank = 1.0 / position
            break

    hits = 0
    precision_sum = 0.0
    for position, place_id in enumerate(top_k, start=1):
        if place_id in relevant_ids:
            hits += 1
            precision_sum += hits / position
    average_precision = precision_sum / len(relevant_ids) if relevant_ids else 0.0

    retrieved_grades = [relevance.get(place_id, 0) for place_id in top_k]
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    dcg = _discounted_cumulative_gain(retrieved_grades)
    ideal_dcg = _discounted_cumulative_gain(ideal_grades)
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return SearchMetricValues(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=reciprocal_rank,
        map=average_precision,
        ndcg_at_k=ndcg,
    )


def average_metrics(values: Sequence[SearchMetricValues]) -> SearchMetricValues:
    if not values:
        return SearchMetricValues(0.0, 0.0, 0.0, 0.0, 0.0)

    count = len(values)
    return SearchMetricValues(
        precision_at_k=sum(item.precision_at_k for item in values) / count,
        recall_at_k=sum(item.recall_at_k for item in values) / count,
        mrr=sum(item.mrr for item in values) / count,
        map=sum(item.map for item in values) / count,
        ndcg_at_k=sum(item.ndcg_at_k for item in values) / count,
    )


def _discounted_cumulative_gain(grades: Sequence[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(grades)
    )
