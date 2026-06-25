from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from app.shared.config.settings import Settings
from app.shared.content_hash import stable_content_hash
from app.shared.nlp.preprocessing.text import clean_text


@dataclass(frozen=True)
class PostSourceRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    content_hash: str
    is_active: bool


class MainApiPostsClient:
    """Reads real posts from the main product API for offline embedding jobs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_posts_search_path.lstrip("/")

    async def iter_posts(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PostSourceRecord]:
        limit = page_limit or self._settings.main_api_posts_page_limit
        page = 1
        offset = 0
        headers = self._build_headers()
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            while True:
                params = self._build_pagination_params(limit, page, offset)
                response = await client.get(self._path, params=params)
                response.raise_for_status()
                posts = self._extract_posts(response.json())
                if not posts:
                    break

                yielded_this_page = 0
                for post in posts:
                    record = post_to_source_record(post)
                    if record is not None and record.id not in seen_ids:
                        seen_ids.add(record.id)
                        yielded_this_page += 1
                        yield record

                if (
                    yielded_this_page == 0
                    or len(posts) < limit
                    or (max_pages is not None and page >= max_pages)
                ):
                    break

                page += 1
                offset += limit

    def _build_headers(self) -> dict[str, str]:
        token = self._settings.main_api_internal_token or self._settings.main_api_auth_token
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _build_pagination_params(
        self,
        limit: int,
        page: int,
        offset: int,
    ) -> dict[str, int]:
        params = {"limit": limit}
        if self._settings.main_api_posts_pagination_mode == "offset":
            params["offset"] = offset
        else:
            params["page"] = page
        return params

    @staticmethod
    def _extract_posts(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "posts", "items", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
            if isinstance(candidate, dict):
                nested = MainApiPostsClient._extract_posts(candidate)
                if nested:
                    return nested
        return []


def post_to_source_record(post: dict[str, Any]) -> PostSourceRecord | None:
    post_id = _first_present(post, "id", "_id", "post_id", "uuid")
    if post_id is None:
        return None

    title = str(_first_present(post, "title", "name", default="")).strip()
    city = _first_present(post, "city", "municipality")
    state = _first_present(post, "state", default="Chiapas")
    text = str(_first_present(post, "text", "content", "description", "body", default=""))
    source = _first_present(post, "source")
    is_active = _first_present(post, "is_active", "isActive", default=True)
    tags = _as_text_list(_first_present(post, "tags", "keywords", default=[]))

    document = clean_text(
        " ".join(
            str(value)
            for value in [title, city, state, source, " ".join(tags), text]
            if value
        )
    )
    metadata = {
        "title": title,
        "city": city,
        "state": state,
        "source": source,
        "tags": ",".join(tags),
        "is_active": bool(is_active),
    }
    filtered_metadata = {
        key: value for key, value in metadata.items() if value not in (None, "")
    }
    content_hash = stable_content_hash(
        {
            "document": document,
            "metadata": filtered_metadata,
            "is_active": bool(is_active),
        }
    )
    return PostSourceRecord(
        id=str(post_id),
        document=document,
        metadata=filtered_metadata,
        content_hash=content_hash,
        is_active=bool(is_active),
    )


def _first_present(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]
