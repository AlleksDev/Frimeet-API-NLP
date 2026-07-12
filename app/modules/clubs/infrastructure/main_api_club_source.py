from typing import Any, AsyncIterator

from app.modules.clubs.infrastructure.semantic_document import (
    CLUB_SEARCH_DOCUMENT_VERSION,
    build_club_search_document,
)
from app.shared.config.settings import Settings
from app.shared.search.documents import first_present, make_search_record
from app.shared.search.source import PagedMainApiSearchClient, SearchSourceRecord


class MainApiClubsClient(PagedMainApiSearchClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings=settings,
            path=settings.main_api_clubs_snapshot_path,
            collection_keys=("clubs",),
            mapper=club_to_source_record,
            page_limit=settings.main_api_search_page_limit,
            pagination_mode=settings.main_api_search_pagination_mode,
        )

    async def iter_clubs(self, **kwargs: Any) -> AsyncIterator[SearchSourceRecord]:
        async for record in self.iter_records(**kwargs):
            yield record


def club_to_source_record(club: dict[str, Any]) -> SearchSourceRecord | None:
    club_id = first_present(club, "id", "_id", "club_id", "uuid")
    if club_id is None:
        return None
    name = str(first_present(club, "name", "title", default="")).strip()
    category = str(first_present(club, "category", "type", default="")).strip()
    description = str(first_present(club, "description", "summary", default="")).strip()
    is_active = bool(first_present(club, "is_active", "isActive", default=True))
    is_private = bool(first_present(club, "is_private", "isPrivate", default=False))
    document = build_club_search_document(name, category, description)
    return make_search_record(
        resource_id=club_id,
        document=document,
        document_version=CLUB_SEARCH_DOCUMENT_VERSION,
        is_active=is_active,
        metadata={
            "search_title": name,
            "name": name,
            "description": description[:300],
            "category": category,
            "is_online": first_present(club, "is_online", "isOnline", default=False),
            "is_private": is_private,
            "place_id": first_present(club, "place_id", "placeId"),
            "image_url": first_present(club, "image_url", "imageUrl"),
            "meeting_schedules": first_present(
                club, "meeting_schedules", "meetingSchedules", default=[]
            ),
            "is_active": is_active,
            "document_version": CLUB_SEARCH_DOCUMENT_VERSION,
        },
    )
