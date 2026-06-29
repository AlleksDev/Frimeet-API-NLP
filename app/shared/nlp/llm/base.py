from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Sequence


PlaceResponseMode = Literal["no_match", "low_confidence", "confident"]


@dataclass(frozen=True)
class LLMResult:
    message: str
    provider: str
    model: str


class LLMProviderError(Exception):
    pass


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    async def generate_place_chat_response(
        self,
        user_intent: str,
        region: str | None,
        places: Sequence[dict[str, Any]],
        response_mode: PlaceResponseMode = "confident",
    ) -> LLMResult:
        """Draft a conversational answer using only provided place candidates."""
