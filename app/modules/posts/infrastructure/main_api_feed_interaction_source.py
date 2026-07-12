from datetime import UTC, datetime
from typing import Any, AsyncIterator

import httpx

from app.modules.posts.domain.profiles import FeedInteraction
from app.shared.config.settings import Settings


class MainApiFeedInteractionSource:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.main_api_base_url.rstrip("/") + "/"
        self._path = settings.main_api_feed_interactions_path.lstrip("/")
        self._timeout = settings.main_api_timeout_seconds
        token = settings.main_api_internal_token
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def iter_interactions(
        self,
        after_id: int,
        page_limit: int = 1000,
        user_id: str | None = None,
    ) -> AsyncIterator[FeedInteraction]:
        current = max(0, after_id)
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, headers=self._headers
        ) as client:
            while True:
                params: dict[str, int | str] = {
                    "after_id": current,
                    "limit": page_limit,
                }
                if user_id is not None:
                    params["user_id"] = user_id
                response = await client.get(self._path, params=params)
                response.raise_for_status()
                payload = response.json()
                items = payload.get("data", []) if isinstance(payload, dict) else []
                if not items:
                    break
                advanced = False
                for raw in items:
                    item = interaction_to_domain(raw)
                    if item.event_id <= current:
                        raise ValueError(
                            "interaction changes debe estar ordenado por event_id "
                            "estrictamente ascendente"
                        )
                    current = item.event_id
                    advanced = True
                    yield item
                if not advanced or not bool(payload.get("has_more", False)):
                    break


def interaction_to_domain(payload: dict[str, Any]) -> FeedInteraction:
    try:
        occurred = datetime.fromisoformat(
            str(payload["occurred_at"]).replace("Z", "+00:00")
        )
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=UTC)
        raw_event_id = payload.get("event_id")
        if raw_event_id is None:
            raw_event_id = payload["id"]
        return FeedInteraction(
            event_id=int(raw_event_id),
            user_id=str(payload["user_id"]),
            post_id=str(payload["post_id"]),
            event_type=str(payload.get("event_type", payload.get("type", ""))).lower(),
            occurred_at=occurred,
            dwell_time_ms=(
                int(payload["dwell_time_ms"])
                if payload.get("dwell_time_ms") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("interaccion invalida recibida de la API principal") from exc
