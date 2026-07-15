from typing import Protocol, Sequence

from app.modules.places.domain.chat_intent import (
    ParsedPlaceChatIntent,
    PlaceChatCandidate,
)


class HybridPlaceChatRetriever(Protocol):
    async def retrieve(
        self,
        intent: ParsedPlaceChatIntent,
        limit: int,
    ) -> Sequence[PlaceChatCandidate]:
        """Retrieve and rank candidates by content only, never by distance."""

