from typing import Protocol, Sequence

from app.modules.places.domain.chat_intent import ResolvedPlaceAnchor


class PlaceAnchorResolver(Protocol):
    async def resolve(
        self,
        text: str,
        city: str | None,
        state: str | None,
        limit: int = 3,
    ) -> Sequence[ResolvedPlaceAnchor]:
        """Resolve a user-facing place name through an authoritative source."""

