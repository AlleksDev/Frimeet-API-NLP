from datetime import UTC, datetime
from typing import Any, AsyncIterator

from app.modules.events.infrastructure.semantic_document import (
    EVENT_SEARCH_DOCUMENT_VERSION,
    build_event_search_document,
)
from app.shared.config.settings import Settings
from app.shared.search.documents import as_text_list, first_present, make_search_record
from app.shared.search.source import PagedMainApiSearchClient, SearchSourceRecord


class MainApiEventsClient(PagedMainApiSearchClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings=settings,
            path=settings.main_api_events_snapshot_path,
            collection_keys=("events",),
            mapper=event_to_source_record,
            page_limit=settings.main_api_search_page_limit,
            pagination_mode=settings.main_api_search_pagination_mode,
        )

    async def iter_events(self, **kwargs: Any) -> AsyncIterator[SearchSourceRecord]:
        async for record in self.iter_records(**kwargs):
            yield record


def event_to_source_record(event: dict[str, Any]) -> SearchSourceRecord | None:
    event_id = first_present(event, "id", "_id", "event_id", "uuid")
    if event_id is None:
        return None
    title = str(first_present(event, "title", "name", default="")).strip()
    description = str(first_present(event, "description", "summary", default="")).strip()
    tag_names = as_text_list(first_present(event, "tags", default=[]))
    tag_ids = as_text_list(first_present(event, "tag_ids", "tagIds", default=[]))
    # Only textual tag names are semantic. UUID/numeric IDs stay as filters/metadata.
    tag_names = [tag for tag in tag_names if not _looks_like_identifier(tag)]
    start_time, duration_minutes = _normalized_temporal_metadata(event)
    temporal_metadata_valid = start_time is not None and duration_minutes is not None
    is_active = (
        bool(first_present(event, "is_active", "isActive", default=True))
        and temporal_metadata_valid
    )
    is_public = bool(first_present(event, "is_public", "isPublic", default=True))
    document = build_event_search_document(title, tag_names, description)
    return make_search_record(
        resource_id=event_id,
        document=document,
        document_version=EVENT_SEARCH_DOCUMENT_VERSION,
        is_active=is_active,
        metadata={
            "search_title": title,
            "title": title,
            "description": description[:300],
            "club_id": first_present(event, "club_id", "clubId"),
            "place_id": first_present(event, "place_id", "placeId"),
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "max_attendees": first_present(event, "max_attendees", "maxAttendees"),
            "is_public": is_public,
            "cover_image_url": first_present(event, "cover_image_url", "coverImageUrl"),
            "tags": tag_names,
            "tag_ids": tag_ids,
            "is_active": is_active,
            "temporal_metadata_valid": temporal_metadata_valid,
            "document_version": EVENT_SEARCH_DOCUMENT_VERSION,
        },
    )


def _looks_like_identifier(value: str) -> bool:
    compact = value.replace("-", "")
    return compact.isdigit() or (len(compact) == 32 and all(c in "0123456789abcdefABCDEF" for c in compact))


def _normalized_temporal_metadata(
    event: dict[str, Any],
) -> tuple[str | None, int | None]:
    raw_start = first_present(event, "start_time", "startTime")
    if not isinstance(raw_start, str) or not raw_start.strip():
        return None, None
    try:
        parsed_start = datetime.fromisoformat(
            raw_start.strip().replace("Z", "+00:00")
        )
    except ValueError:
        return None, None
    if parsed_start.utcoffset() is None:
        return None, None

    raw_duration = first_present(event, "duration_minutes", "durationMinutes")
    if isinstance(raw_duration, bool):
        return None, None
    try:
        parsed_duration = int(raw_duration)
    except (TypeError, ValueError):
        return None, None
    if parsed_duration <= 0 or parsed_duration > 2_147_483_647:
        return None, None
    return parsed_start.astimezone(UTC).isoformat(), parsed_duration
