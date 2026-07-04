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
            path=settings.main_api_events_search_path,
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
    tag_values = first_present(event, "tags", "tag_ids", "tagIds", default=[])
    tag_ids = as_text_list(tag_values)
    # Only textual tag names are semantic. UUID/numeric IDs stay as filters/metadata.
    tag_names = [tag for tag in tag_ids if not _looks_like_identifier(tag)]
    is_active = bool(first_present(event, "is_active", "isActive", default=True))
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
            "start_time": first_present(event, "start_time", "startTime"),
            "duration_minutes": first_present(
                event, "duration_minutes", "durationMinutes"
            ),
            "max_attendees": first_present(event, "max_attendees", "maxAttendees"),
            "is_public": is_public,
            "cover_image_url": first_present(event, "cover_image_url", "coverImageUrl"),
            "tags": tag_names,
            "tag_ids": tag_ids,
            "is_active": is_active,
            "document_version": EVENT_SEARCH_DOCUMENT_VERSION,
        },
    )


def _looks_like_identifier(value: str) -> bool:
    compact = value.replace("-", "")
    return compact.isdigit() or (len(compact) == 32 and all(c in "0123456789abcdefABCDEF" for c in compact))
