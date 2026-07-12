from datetime import UTC, datetime
from typing import Any, AsyncIterator

import httpx

from app.shared.config.settings import Settings
from app.shared.content_hash import stable_content_hash
from app.shared.nlp.preprocessing.text import clean_text
from app.modules.posts.domain.sync import PostChangeRecord, PostSourceRecord


class MainApiPostsClient:
    """Reads real posts from the main product API for offline embedding jobs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._snapshot_path = settings.main_api_posts_snapshot_path.lstrip("/")
        self._changes_path = settings.main_api_posts_changes_path.lstrip("/")

    async def iter_snapshot(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PostSourceRecord]:
        limit = page_limit or self._settings.main_api_posts_page_limit
        page = 1
        offset = 0
        cursor: str | None = None
        headers = self._build_headers()
        seen_ids: set[str] = set()
        pagination_mode = self._settings.main_api_posts_pagination_mode

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            while True:
                params = self._build_pagination_params(limit, page, offset, cursor)
                response = await client.get(self._snapshot_path, params=params)
                response.raise_for_status()
                payload = response.json()
                posts = self._extract_posts(payload)
                if not posts:
                    break

                yielded_this_page = 0
                for post in posts:
                    record = post_to_source_record(post)
                    if record is not None and record.id not in seen_ids:
                        seen_ids.add(record.id)
                        yielded_this_page += 1
                        yield record

                if yielded_this_page == 0 or (max_pages is not None and page >= max_pages):
                    break

                if pagination_mode == "cursor":
                    cursor = self._extract_next_cursor(payload)
                    if not self._extract_has_more(payload) or not cursor:
                        break
                elif len(posts) < limit:
                    break

                page += 1
                offset += limit

    def iter_posts(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PostSourceRecord]:
        """Backward-compatible alias for the snapshot iterator."""
        return self.iter_snapshot(page_limit, max_pages)

    async def iter_changes(
        self,
        after_id: int,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[PostChangeRecord]:
        limit = page_limit or self._settings.main_api_posts_page_limit
        page = 1
        current_after_id = max(0, after_id)
        headers = self._build_headers()
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=headers,
        ) as client:
            while True:
                response = await client.get(
                    self._changes_path,
                    params={"after_id": current_after_id, "limit": limit},
                )
                response.raise_for_status()
                payload = response.json()
                items = self._extract_posts(payload)
                if not items:
                    break
                last_event_id = current_after_id
                for item in items:
                    change = post_change_to_record(item)
                    if change is None:
                        raise ValueError("cambio de post invalido recibido de la API principal")
                    if change.event_id <= last_event_id:
                        raise ValueError(
                            "post changes debe estar ordenado por event_id "
                            "estrictamente ascendente"
                        )
                    last_event_id = change.event_id
                    yield change
                if last_event_id == current_after_id:
                    break
                current_after_id = last_event_id
                if not self._extract_has_more(payload):
                    break
                if max_pages is not None and page >= max_pages:
                    break
                page += 1

    def _build_headers(self) -> dict[str, str]:
        token = self._settings.main_api_internal_token
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _build_pagination_params(
        self,
        limit: int,
        page: int,
        offset: int,
        cursor: str | None,
    ) -> dict[str, int | str]:
        params: dict[str, int | str] = {"limit": limit}
        mode = self._settings.main_api_posts_pagination_mode
        if mode == "cursor":
            if cursor:
                params["cursor"] = cursor
        elif mode == "offset":
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

    @staticmethod
    def _extract_next_cursor(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        cursor = payload.get("next_cursor") or payload.get("nextCursor")
        if cursor is None:
            return None
        cursor = str(cursor).strip()
        return cursor or None

    @staticmethod
    def _extract_has_more(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        value = payload.get("has_more", payload.get("hasMore", False))
        return bool(value)


def post_to_source_record(post: dict[str, Any]) -> PostSourceRecord | None:
    post_id = _first_present(post, "id", "_id", "post_id", "uuid")
    if post_id is None:
        return None

    title = str(_first_present(post, "title", "name", default="")).strip()
    city = _first_present(post, "city", "municipality")
    state = _first_present(post, "state", default="Chiapas")
    text = str(_first_present(post, "text", "content", "description", "body", default=""))
    source = _first_present(post, "source")
    published_at = _first_present(
        post, "published_at", "publishedAt", "created_at", "createdAt"
    )
    is_active = _first_present(post, "is_active", "isActive", default=True)
    author_type = _first_present(post, "author_type", "authorType")
    author_id = _first_present(post, "author_id", "authorId")
    if author_id is None and isinstance(post.get("author"), dict):
        author_type = author_type or post["author"].get("type")
        author_id = post["author"].get("id")
    source_version = _as_int(_first_present(post, "source_version", "sourceVersion"))
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
        "published_at": published_at,
        "tags": ",".join(tags),
        "is_active": bool(is_active),
        "author_type": author_type,
        "author_id": author_id,
        "source_version": source_version,
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
        author_type=str(author_type) if author_type is not None else None,
        author_id=str(author_id) if author_id is not None else None,
        published_at=_as_datetime(published_at),
        source_version=source_version,
    )


def post_change_to_record(payload: dict[str, Any]) -> PostChangeRecord | None:
    event_id = _as_int(_first_present(payload, "event_id", "id", "outbox_id"))
    post_id = _first_present(payload, "post_id", "aggregate_id")
    operation = str(_first_present(payload, "operation", "event_type", default="upsert")).lower()
    source_version = _as_int(
        _first_present(payload, "source_version", "version", default=event_id)
    )
    if event_id is None or post_id is None or source_version is None:
        return None
    normalized = {
        "post.created": "upsert",
        "post.updated": "upsert",
        "post.restored": "upsert",
        "post.archived": "archive",
        "post.deleted": "delete",
    }.get(operation, operation)
    raw_post = payload.get("post") or payload.get("data")
    if raw_post is None and normalized == "upsert":
        raw_post = payload
    post = post_to_source_record(raw_post) if isinstance(raw_post, dict) else None
    return PostChangeRecord(event_id, str(post_id), normalized, source_version, post)


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


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None
