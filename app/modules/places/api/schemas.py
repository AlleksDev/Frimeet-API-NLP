from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.modules.places.domain.models import PlaceCandidate, PlaceFilters


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


class PlaceSearchResponse(BaseModel):
    query: str
    places: list[PlaceResultSchema]


class PlaceRecommendationResponse(BaseModel):
    query: str
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
