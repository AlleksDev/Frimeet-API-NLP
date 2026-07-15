from typing import Protocol

from app.modules.places.domain.chat_intent import (
    ConversationState,
    ParsedPlaceChatIntent,
)


class PlaceChatIntentParser(Protocol):
    def parse(
        self,
        message: str,
        state: ConversationState,
        has_user_location: bool,
    ) -> ParsedPlaceChatIntent:
        """Parse one turn into a structured, non-authoritative intent."""

