from typing import Any, AsyncIterator

from app.modules.users.infrastructure.semantic_document import (
    USER_SEARCH_DOCUMENT_VERSION,
    build_user_search_document,
)
from app.shared.config.settings import Settings
from app.shared.search.documents import first_present, make_search_record
from app.shared.search.source import PagedMainApiSearchClient, SearchSourceRecord


class MainApiUsersClient(PagedMainApiSearchClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings=settings,
            path=settings.main_api_users_search_path,
            collection_keys=("users",),
            mapper=user_to_source_record,
            page_limit=settings.main_api_search_page_limit,
            pagination_mode=settings.main_api_search_pagination_mode,
        )

    async def iter_users(self, **kwargs: Any) -> AsyncIterator[SearchSourceRecord]:
        async for record in self.iter_records(**kwargs):
            yield record


def user_to_source_record(user: dict[str, Any]) -> SearchSourceRecord | None:
    user_id = first_present(user, "id", "_id", "user_id", "uuid")
    if user_id is None:
        return None
    username = str(first_present(user, "username", default="")).strip()
    full_name = str(first_present(user, "full_name", "fullName", "name", default="")).strip()
    bio = str(first_present(user, "bio", "biography", default="")).strip()
    is_active = bool(first_present(user, "is_active", "isActive", default=True))
    document = build_user_search_document(username, full_name, bio)
    return make_search_record(
        resource_id=user_id,
        document=document,
        document_version=USER_SEARCH_DOCUMENT_VERSION,
        is_active=is_active,
        metadata={
            "search_title": full_name or username,
            "username": username,
            "full_name": full_name,
            "avatar_url": first_present(user, "avatar_url", "avatarUrl"),
            "bio": bio[:300],
            "role": first_present(user, "role"),
            "is_active": is_active,
            "document_version": USER_SEARCH_DOCUMENT_VERSION,
        },
    )
