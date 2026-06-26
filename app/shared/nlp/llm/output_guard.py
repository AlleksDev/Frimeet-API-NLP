from dataclasses import dataclass
import re


DEFAULT_PLACE_CHAT_FALLBACK = (
    "Encontré algunas opciones que coinciden con lo que buscas. "
    "Puedes revisar sus detalles y elegir la que mejor se adapte a tu plan."
)


@dataclass(frozen=True)
class GuardedLLMMessage:
    message: str
    used_fallback: bool
    reason: str | None = None


class PlaceChatOutputGuard:
    def __init__(self, max_chars: int = 700) -> None:
        self._max_chars = max_chars

    def validate(
        self,
        message: str,
        allowed_place_names: list[str],
    ) -> GuardedLLMMessage:
        cleaned = " ".join((message or "").split())
        if len(cleaned) < 10:
            return self.fallback("empty_or_too_short")

        if self._contains_unsupported_claims(cleaned):
            return self.fallback("unsupported_claims")

        if len(cleaned) > self._max_chars:
            cleaned = cleaned[: self._max_chars].rsplit(" ", 1)[0] + "..."

        allowed_names = [name.casefold() for name in allowed_place_names]
        if allowed_names and self._mentions_explicit_place(cleaned):
            lowered = cleaned.casefold()
            has_allowed_name = any(name in lowered for name in allowed_names)
            if not has_allowed_name:
                return self.fallback("mentions_unverified_place")

        return GuardedLLMMessage(message=cleaned, used_fallback=False)

    def fallback(self, reason: str | None = None) -> GuardedLLMMessage:
        return GuardedLLMMessage(
            message=DEFAULT_PLACE_CHAT_FALLBACK,
            used_fallback=True,
            reason=reason,
        )

    @staticmethod
    def _contains_unsupported_claims(message: str) -> bool:
        patterns = [
            r"\babiert[oa]s?\b",
            r"\bpromoci[oó]n\b",
            r"\bdescuento\b",
            r"\bgratis\b",
            r"\$\s?\d+",
            r"\b\d{1,2}:\d{2}\b",
        ]
        return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _mentions_explicit_place(message: str) -> bool:
        return bool(
            re.search(
                r"\b(lugar|opci[oó]n|restaurante|caf[eé]|mirador)\s+[A-ZÁÉÍÓÚÑ]",
                message,
            )
        )
