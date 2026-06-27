from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_metrics import (
    SearchEngineMetrics,
    SearchMetricValues,
)


class PlaceFiltersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=80)
    price_range: str | None = Field(default=None, max_length=30)
    is_active: bool | None = True
    occasion: str | None = Field(default=None, max_length=80)


class PlaceSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    filters: PlaceFiltersSchema = Field(default_factory=PlaceFiltersSchema)
    limit: int = Field(default=10, ge=1, le=20)

    def to_domain_filters(self) -> PlaceFilters:
        return PlaceFilters(
            city=self.city or self.filters.city,
            state=self.state or self.filters.state,
            category=self.filters.category,
            price_range=self.filters.price_range,
            is_active=self.filters.is_active,
            occasion=self.filters.occasion,
        )


class PlaceRecommendationRequest(PlaceSearchRequest):
    pass


class PlaceChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=1000)
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    filters: PlaceFiltersSchema = Field(default_factory=PlaceFiltersSchema)
    limit: int = Field(default=5, ge=1, le=8)

    def to_domain_filters(self) -> PlaceFilters:
        return PlaceFilters(
            city=self.city or self.filters.city,
            state=self.state or self.filters.state,
            category=self.filters.category,
            price_range=self.filters.price_range,
            is_active=self.filters.is_active,
            occasion=self.filters.occasion,
        )


class PlaceResultSchema(BaseModel):
    id: str
    name: str
    score: float
    category: str | None = None
    city: str | None = None
    state: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlaceSearchEngineMetricsSchema(BaseModel):
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


class PlaceSearchResponse(BaseModel):
    query: str
    places: list[PlaceResultSchema]
    metrics: PlaceSearchEngineMetricsSchema


class PlaceSearchEvaluationCaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    relevance: dict[str, int] = Field(..., min_length=1)
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    filters: PlaceFiltersSchema = Field(default_factory=PlaceFiltersSchema)

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, value: dict[str, int]) -> dict[str, int]:
        if any(grade < 1 or grade > 3 for grade in value.values()):
            raise ValueError("relevance grades must be between 1 and 3")
        return value

    def to_domain_filters(self) -> PlaceFilters:
        return PlaceFilters(
            city=self.city or self.filters.city,
            state=self.state or self.filters.state,
            category=self.filters.category,
            price_range=self.filters.price_range,
            is_active=self.filters.is_active,
            occasion=self.filters.occasion,
        )


class PlaceSearchMetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[PlaceSearchEvaluationCaseSchema] = Field(
        ...,
        min_length=1,
        max_length=50,
    )
    k: int = Field(default=5, ge=1, le=20)


class PlaceSearchMetricValuesSchema(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    map: float
    ndcg_at_k: float


class PlaceSearchQueryMetricsSchema(BaseModel):
    query: str
    ranking: list[str]
    metrics: PlaceSearchMetricValuesSchema


class PlaceSearchMetricDefinitionSchema(BaseModel):
    label: str
    description: str


class PlaceSearchRecommendedMetricSchema(BaseModel):
    key: str
    label: str
    value: float
    rationale: str


class PlaceSearchMetricsResponse(BaseModel):
    engine: str
    k: int
    query_count: int
    metric_definitions: dict[str, PlaceSearchMetricDefinitionSchema]
    recommended_metric: PlaceSearchRecommendedMetricSchema
    aggregate: PlaceSearchMetricValuesSchema
    queries: list[PlaceSearchQueryMetricsSchema]


class PlaceRecommendationResponse(BaseModel):
    query: str
    message: str
    places: list[PlaceResultSchema]
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlaceChatResponse(BaseModel):
    response_id: str
    nlp_trace_id: str
    message: str
    places: list[PlaceResultSchema]
    metadata: dict[str, Any]


def place_to_schema(place: PlaceCandidate) -> PlaceResultSchema:
    return PlaceResultSchema(
        id=place.id,
        name=place.name,
        score=round(place.score, 4),
        category=place.category,
        city=place.city,
        state=place.state,
        metadata=place.metadata,
    )


def metric_values_to_schema(
    metrics: SearchMetricValues,
) -> PlaceSearchMetricValuesSchema:
    return PlaceSearchMetricValuesSchema(
        precision_at_k=round(metrics.precision_at_k, 4),
        recall_at_k=round(metrics.recall_at_k, 4),
        mrr=round(metrics.mrr, 4),
        map=round(metrics.map, 4),
        ndcg_at_k=round(metrics.ndcg_at_k, 4),
    )


def metric_definitions_to_schema(
    k: int,
) -> dict[str, PlaceSearchMetricDefinitionSchema]:
    return {
        "precision_at_k": PlaceSearchMetricDefinitionSchema(
            label=f"Precision@{k}",
            description="Proporcion del top-k que realmente es relevante.",
        ),
        "recall_at_k": PlaceSearchMetricDefinitionSchema(
            label=f"Recall@{k}",
            description="Proporcion de todos los relevantes recuperada en el top-k.",
        ),
        "mrr": PlaceSearchMetricDefinitionSchema(
            label="MRR",
            description="Premia que el primer resultado relevante aparezca pronto.",
        ),
        "map": PlaceSearchMetricDefinitionSchema(
            label="MAP",
            description="Promedia la precision en las posiciones relevantes.",
        ),
        "ndcg_at_k": PlaceSearchMetricDefinitionSchema(
            label=f"nDCG@{k}",
            description="Evalua orden y relevancia graduada dentro del top-k.",
        ),
    }


def recommended_metric_to_schema(
    metrics: SearchMetricValues,
    k: int,
) -> PlaceSearchRecommendedMetricSchema:
    return PlaceSearchRecommendedMetricSchema(
        key="ndcg_at_k",
        label=f"nDCG@{k}",
        value=round(metrics.ndcg_at_k, 4),
        rationale=(
            "Es la mejor metrica principal para una lista de lugares porque considera "
            "la posicion y los grados de relevancia."
        ),
    )


def engine_metrics_to_schema(
    metrics: SearchEngineMetrics,
) -> PlaceSearchEngineMetricsSchema:
    return PlaceSearchEngineMetricsSchema(
        engine=metrics.engine,
        candidate_retrieval=metrics.candidate_retrieval,
        score_metric=metrics.score_metric,
        field_weights=metrics.field_weights,
        candidate_count=metrics.candidate_count,
        returned_count=metrics.returned_count,
        nonzero_score_count=metrics.nonzero_score_count,
        min_score=round(metrics.min_score, 4),
        max_score=round(metrics.max_score, 4),
        mean_score=round(metrics.mean_score, 4),
    )
