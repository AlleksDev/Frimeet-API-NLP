from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from app.shared.config.settings import Settings


@dataclass(frozen=True)
class SearchSourceRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    content_hash: str
    is_active: bool = True


RecordMapper = Callable[[dict[str, Any]], SearchSourceRecord | None]


class PagedMainApiSearchClient:
    """Generic paginated reader; each resource still owns its record mapper."""

    def __init__(
        self,
        settings: Settings,
        path: str,
        collection_keys: tuple[str, ...],
        mapper: RecordMapper,
        page_limit: int,
        pagination_mode: str,
    ) -> None:
        self._settings = settings
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = path.lstrip("/")
        if not self._path.startswith("api/v1/internal/search/"):
            raise ValueError("search sync source must use an internal snapshot endpoint")
        self._collection_keys = collection_keys
        self._mapper = mapper
        self._page_limit = page_limit
        self._pagination_mode = pagination_mode

    async def iter_records(
        self,
        page_limit: int | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[SearchSourceRecord]:
        limit = page_limit or self._page_limit
        page = 1
        offset = 0
        cursor: str | None = None
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.main_api_timeout_seconds,
            headers=self._build_headers(),
        ) as client:
            while True:
                response = await client.get(
                    self._path,
                    params=self._pagination_params(limit, page, offset, cursor),
                )
                response.raise_for_status()
                payload = response.json()
                resources = self._extract_resources(payload)
                if not resources:
                    break

                yielded = 0
                for resource in resources:
                    record = self._mapper(resource)
                    if record is not None and record.id not in seen_ids:
                        seen_ids.add(record.id)
                        yielded += 1
                        yield record

                if yielded == 0 or (max_pages is not None and page >= max_pages):
                    break

                if self._pagination_mode == "cursor":
                    cursor = self._extract_next_cursor(payload)
                    if not self._extract_has_more(payload) or not cursor:
                        break
                elif len(resources) < limit:
                    break

                page += 1
                offset += limit

    def _build_headers(self) -> dict[str, str]:
        token = (self._settings.main_api_internal_token or "").strip()
        if not token:
            raise RuntimeError(
                "MAIN_API_INTERNAL_TOKEN is required for search snapshot synchronization"
            )
        return {"Authorization": f"Bearer {token}"}

    def _pagination_params(
        self,
        limit: int,
        page: int,
        offset: int,
        cursor: str | None,
    ) -> dict[str, int | str]:
        params: dict[str, int | str] = {"limit": limit}
        if self._pagination_mode == "cursor":
            if cursor:
                params["cursor"] = cursor
        elif self._pagination_mode == "offset":
            params["offset"] = offset
        else:
            params["page"] = page
        return params

    def _extract_resources(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            resources = [item for item in payload if isinstance(item, dict)]
            if resources:
                return resources
            return []
        if not isinstance(payload, dict):
            return []

        preferred_keys = ("data", *self._collection_keys, "items", "results")
        for key in preferred_keys:
            candidate = payload.get(key)
            nested = self._extract_resources(candidate)
            if nested:
                return nested

        # Some main-API endpoints add an extra response-envelope key. Search
        # nested containers as a fallback without coupling the indexer to it.
        for key, candidate in payload.items():
            if key in preferred_keys or not isinstance(candidate, (dict, list)):
                continue
            nested = self._extract_resources(candidate)
            if nested:
                return nested
        return []

    @staticmethod
    def _extract_next_cursor(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        cursor = payload.get("next_cursor") or payload.get("nextCursor")
        return str(cursor).strip() or None if cursor is not None else None

    @staticmethod
    def _extract_has_more(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("has_more", payload.get("hasMore", False)))
