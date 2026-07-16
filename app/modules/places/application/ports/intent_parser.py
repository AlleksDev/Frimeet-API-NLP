from typing import Protocol

from app.modules.places.domain.chat_intent import (
    ClarificationChoice,
    ConversationState,
    ParsedPlaceChatIntent,
)


class PlaceChatIntentParser(Protocol):
    def parse(
        self,
        message: str,
        state: ConversationState,
        has_user_location: bool,
        clarification_choice: ClarificationChoice | None = None,
    ) -> ParsedPlaceChatIntent:
        """Parse one turn into a structured, non-authoritative intent."""
