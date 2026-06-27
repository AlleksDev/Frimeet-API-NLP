from dataclasses import dataclass
from typing import Sequence

from app.modules.places.application.use_cases.search_places import SearchPlacesUseCase
from app.modules.places.domain.models import PlaceFilters
from app.modules.places.domain.search_metrics import (
    SearchMetricValues,
    average_metrics,
    evaluate_ranking,
)


@dataclass(frozen=True)
class PlaceSearchEvaluationCase:
    query: str
    relevance: dict[str, int]
    filters: PlaceFilters


@dataclass(frozen=True)
class PlaceSearchQueryMetrics:
    query: str
    ranking: list[str]
    metrics: SearchMetricValues


@dataclass(frozen=True)
class EvaluatePlaceSearchResult:
    engine: str
    benchmark: str
    qrels_source: str
    k: int
    query_count: int
    aggregate: SearchMetricValues
    queries: list[PlaceSearchQueryMetrics]


class EvaluatePlaceSearchUseCase:
    def __init__(
        self,
        search_use_case: SearchPlacesUseCase,
        cases: Sequence[PlaceSearchEvaluationCase],
        benchmark: str,
        qrels_source: str,
    ) -> None:
        self._search_use_case = search_use_case
        self._cases = tuple(cases)
        self._benchmark = benchmark
        self._qrels_source = qrels_source
        self._result_cache: dict[int, EvaluatePlaceSearchResult] = {}

    async def execute(
        self,
        k: int = 5,
    ) -> EvaluatePlaceSearchResult:
        cached = self._result_cache.get(k)
        if cached is not None:
            return cached

        query_results: list[PlaceSearchQueryMetrics] = []

        for case in self._cases:
            search_result = await self._search_use_case.execute(
                query=case.query,
                filters=case.filters,
                limit=k,
            )
            ranking = [place.id for place in search_result.places]
            query_results.append(
                PlaceSearchQueryMetrics(
                    query=case.query,
                    ranking=ranking,
                    metrics=evaluate_ranking(ranking, case.relevance, k),
                )
            )

        aggregate = average_metrics([result.metrics for result in query_results])
        result = EvaluatePlaceSearchResult(
            engine="tfidf_cosine",
            benchmark=self._benchmark,
            qrels_source=self._qrels_source,
            k=k,
            query_count=len(query_results),
            aggregate=aggregate,
            queries=query_results,
        )
        self._result_cache[k] = result
        return result
