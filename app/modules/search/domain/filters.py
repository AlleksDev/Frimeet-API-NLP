from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Sequence

from app.modules.search.domain.models import SearchHit, SearchResourceType


class LocationSearchMode(StrEnum):
    PRIORITIZE = "prioritize"
    STRICT = "strict"


class ClubAttendanceMode(StrEnum):
    ONLINE = "online"
    IN_PERSON = "in_person"


@dataclass(frozen=True)
class SearchLocation:
    latitude: float
    longitude: float
    radius_meters: int = 5000
    mode: LocationSearchMode = LocationSearchMode.PRIORITIZE


@dataclass(frozen=True)
class GlobalSearchFilters:
    city: str | None = None
    state: str | None = None
    categories: tuple[str, ...] = ()
    price_ranges: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    published_from: datetime | None = None
    published_to: datetime | None = None
    event_from: datetime | None = None
    event_to: datetime | None = None
    club_mode: ClubAttendanceMode | None = None
    user_roles: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not any(
            (
                self.city,
                self.state,
                self.categories,
                self.price_ranges,
                self.tags,
                self.published_from,
                self.published_to,
                self.event_from,
                self.event_to,
                self.club_mode,
                self.user_roles,
            )
        )


@dataclass(frozen=True)
class SearchCriteria:
    filters: GlobalSearchFilters = field(default_factory=GlobalSearchFilters)
    location: SearchLocation | None = None
    nearby_place_ids: frozenset[str] = frozenset()
    nearby_boost: float = 0.12

    def with_nearby_place_ids(self, place_ids: set[str]) -> "SearchCriteria":
        return replace(self, nearby_place_ids=frozenset(place_ids))

    def requires_extended_candidates(self, resource_type: SearchResourceType) -> bool:
        if not self.filters.is_empty():
            return True
        return self.location is not None and resource_type in _LOCATION_AWARE_RESOURCES


_LOCATION_AWARE_RESOURCES = {
    SearchResourceType.PLACES,
    SearchResourceType.CLUBS,
    SearchResourceType.EVENTS,
}
def filter_and_rank_hits(
    resource_type: SearchResourceType,
    hits: Sequence[SearchHit],
    criteria: SearchCriteria,
) -> list[SearchHit]:
    accepted: list[SearchHit] = []
    for hit in hits:
        if not _matches_filters(resource_type, hit.metadata, criteria.filters):
            continue
        is_nearby = _is_nearby(resource_type, hit, criteria.nearby_place_ids)
        location_applicable = resource_type in _LOCATION_AWARE_RESOURCES
        if (
            criteria.location is not None
            and criteria.location.mode == LocationSearchMode.STRICT
            and location_applicable
            and not is_nearby
        ):
            continue
        boost = (
            criteria.nearby_boost
            if criteria.location is not None
            and criteria.location.mode == LocationSearchMode.PRIORITIZE
            and is_nearby
            else 0.0
        )
        accepted.append(
            replace(
                hit,
                ranking_score=hit.score + boost,
                is_nearby=(
                    is_nearby
                    if criteria.location is not None and location_applicable
                    else None
                ),
                proximity_boost=boost,
            )
        )
    return sorted(
        accepted,
        key=lambda hit: (
            -(hit.ranking_score if hit.ranking_score is not None else hit.score),
            -hit.score,
            hit.id,
        ),
    )


def _matches_filters(
    resource_type: SearchResourceType,
    metadata: dict[str, Any],
    filters: GlobalSearchFilters,
) -> bool:
    if resource_type in {SearchResourceType.PLACES, SearchResourceType.POSTS}:
        if filters.city and not _same_text(metadata.get("city"), filters.city):
            return False
        if filters.state and not _same_text(metadata.get("state"), filters.state):
            return False

    if resource_type in {SearchResourceType.PLACES, SearchResourceType.CLUBS}:
        if filters.categories and not _contains_text(filters.categories, metadata.get("category")):
            return False

    if resource_type == SearchResourceType.PLACES:
        if filters.price_ranges and not _contains_text(
            filters.price_ranges, metadata.get("price_range")
        ):
            return False

    if filters.tags and resource_type in {
        SearchResourceType.PLACES,
        SearchResourceType.POSTS,
        SearchResourceType.EVENTS,
    }:
        metadata_tags = _as_text_set(metadata.get("tags"))
        if not metadata_tags.intersection(_normalized_set(filters.tags)):
            return False

    if resource_type == SearchResourceType.POSTS and not _date_in_range(
        metadata.get("published_at") or metadata.get("created_at"),
        filters.published_from,
        filters.published_to,
    ):
        return False

    if resource_type == SearchResourceType.EVENTS and not _date_in_range(
        metadata.get("start_time"), filters.event_from, filters.event_to
    ):
        return False

    if resource_type == SearchResourceType.CLUBS and filters.club_mode is not None:
        expected_online = filters.club_mode == ClubAttendanceMode.ONLINE
        if bool(metadata.get("is_online", False)) != expected_online:
            return False

    if resource_type == SearchResourceType.USERS and filters.user_roles:
        if not _contains_text(filters.user_roles, metadata.get("role")):
            return False

    return True


def _is_nearby(
    resource_type: SearchResourceType,
    hit: SearchHit,
    nearby_place_ids: frozenset[str],
) -> bool:
    if not nearby_place_ids:
        return False
    place_id = hit.id if resource_type == SearchResourceType.PLACES else hit.metadata.get("place_id")
    return place_id is not None and str(place_id) in nearby_place_ids


def _date_in_range(
    raw_value: Any,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if start is None and end is None:
        return True
    parsed = _parse_datetime(raw_value)
    if parsed is None:
        return False
    if start is not None and parsed < _normalized_datetime(start):
        return False
    if end is not None and parsed > _normalized_datetime(end):
        return False
    return True


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _normalized_datetime(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _normalized_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _same_text(left: Any, right: str) -> bool:
    return str(left or "").strip().casefold() == right.strip().casefold()


def _contains_text(values: Sequence[str], candidate: Any) -> bool:
    return str(candidate or "").strip().casefold() in _normalized_set(values)


def _normalized_set(values: Sequence[str]) -> set[str]:
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def _as_text_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).split(",")
    return _normalized_set([str(item) for item in values])
