from typing import Any, AsyncIterator

from app.modules.groups.infrastructure.semantic_document import (
    GROUP_SEARCH_DOCUMENT_VERSION,
    build_group_search_document,
)
from app.shared.config.settings import Settings
from app.shared.search.documents import as_text_list, first_present, make_search_record
from app.shared.search.source import PagedMainApiSearchClient, SearchSourceRecord


class MainApiGroupsClient(PagedMainApiSearchClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings=settings,
            path=settings.main_api_groups_search_path,
            collection_keys=("groups",),
            mapper=group_to_source_record,
            page_limit=settings.main_api_search_page_limit,
            pagination_mode=settings.main_api_search_pagination_mode,
        )

    async def iter_groups(self, **kwargs: Any) -> AsyncIterator[SearchSourceRecord]:
        async for record in self.iter_records(**kwargs):
            yield record


def group_to_source_record(group: dict[str, Any]) -> SearchSourceRecord | None:
    group_id = first_present(group, "id", "_id", "group_id", "uuid")
    if group_id is None:
        return None
    creator_id = str(first_present(group, "creator_id", "creatorId", default="")).strip()
    added_ids = as_text_list(first_present(group, "added_ids", "addedIds", default=[]))
    invited_ids = as_text_list(first_present(group, "invited_ids", "invitedIds", default=[]))
    authorized_user_ids = list(dict.fromkeys([creator_id, *added_ids, *invited_ids]))
    authorized_user_ids = [value for value in authorized_user_ids if value]
    name = str(first_present(group, "name", "title", default="")).strip()
    description = str(first_present(group, "description", "summary", default="")).strip()
    is_active = bool(first_present(group, "is_active", "isActive", default=True))
    is_private = bool(first_present(group, "is_private", "isPrivate", default=True))
    document = build_group_search_document(name, description)
    return make_search_record(
        resource_id=group_id,
        document=document,
        document_version=GROUP_SEARCH_DOCUMENT_VERSION,
        is_active=is_active,
        metadata={
            "search_title": name,
            "name": name,
            "description": description[:300],
            "creator_id": creator_id,
            "image_url": first_present(group, "image_url", "imageUrl"),
            "is_private": is_private,
            "is_pinned": first_present(group, "is_pinned", "isPinned", default=False),
            "member_count": first_present(group, "member_count", "memberCount", default=0),
            "authorized_user_ids": authorized_user_ids,
            "created_at": first_present(group, "created_at", "createdAt"),
            "is_active": is_active,
            "document_version": GROUP_SEARCH_DOCUMENT_VERSION,
        },
    )
