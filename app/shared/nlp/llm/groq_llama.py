import asyncio
from typing import Any, Sequence

from app.shared.config.settings import Settings
from app.shared.nlp.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResult,
    PlaceResponseMode,
)
from app.shared.nlp.prompts.place_chat import build_place_chat_messages


class GroqLlamaProvider(LLMProvider):
    provider_name = "groq"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = settings.groq_model
        self._semaphore = asyncio.Semaphore(settings.max_llm_concurrent_requests)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if not self._settings.groq_api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured")
        if self._client is None:
            try:
                from groq import AsyncGroq
            except ImportError as exc:
                raise LLMProviderError("groq dependency is not installed") from exc
            self._client = AsyncGroq(api_key=self._settings.groq_api_key)
        return self._client

    async def generate_place_chat_response(
        self,
        user_intent: str,
        region: str | None,
        places: Sequence[dict[str, Any]],
        response_mode: PlaceResponseMode = "confident",
    ) -> LLMResult:
        messages = build_place_chat_messages(
            user_intent=user_intent,
            region=region,
            places=places,
            response_mode=response_mode,
        )
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._settings.llm_timeout_seconds):
                    completion = await self._get_client().chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.4,
                        max_tokens=260,
                    )
            except Exception as exc:
                raise LLMProviderError(str(exc)) from exc

        message = completion.choices[0].message.content or ""
        return LLMResult(
            message=message.strip(),
            provider=self.provider_name,
            model=self.model_name,
        )
