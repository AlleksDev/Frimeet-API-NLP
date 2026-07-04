from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SearchResourceType(StrEnum):
    PLACES = "places"
    POSTS = "posts"
    USERS = "users"
    CLUBS = "clubs"
    GROUPS = "groups"
    EVENTS = "events"


ALL_SEARCH_RESOURCE_TYPES = tuple(SearchResourceType)


@dataclass(frozen=True)
class SearchHit:
    id: str
    resource_type: SearchResourceType
    title: str
    score: float
    subtitle: str | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchSectionPagination:
    page_size: int
    returned_count: int
    has_more: bool
    next_offset: int | None = None


@dataclass(frozen=True)
class SearchAllResult:
    query: str
    normalized_query: str
    top_results: list[SearchHit]
    sections: dict[SearchResourceType, list[SearchHit]]
    pagination: dict[SearchResourceType, SearchSectionPagination]
    failed_resources: dict[SearchResourceType, str]
