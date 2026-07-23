from dataclasses import dataclass
import re


DEFAULT_PLACE_CHAT_FALLBACK = (
    "¡Claro! Encontré algunas opciones que se acercan bastante a lo que tienes "
    "en mente. Ojalá alguna se convierta en un buen plan para ti."
)
NO_MATCH_PLACE_CHAT_FALLBACK = (
    "Esta vez no encontré un lugar que encaje bien con lo que buscas. "
    "Si quieres, probamos en otra zona o con una idea parecida."
)
LOW_CONFIDENCE_PLACE_CHAT_FALLBACK = (
    "Encontré algunas posibilidades cercanas a tu idea, aunque todavía no tengo "
    "señales suficientes para asegurarte que sean justo lo que buscas. "
    "Cuéntame qué detalle no puede faltar y lo afinamos juntos."
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
        response_mode: str = "confident",
    ) -> GuardedLLMMessage:
        raw_message = message or ""
        cleaned = " ".join(raw_message.split())
        if len(cleaned) < 10:
            return self.fallback("empty_or_too_short", response_mode)

        if self._contains_meta_or_non_plain_response(raw_message, cleaned):
            return self.fallback("meta_or_non_plain_response", response_mode)

        if self._contains_unsupported_claims(cleaned):
            return self.fallback("unsupported_claims", response_mode)

        if len(cleaned) > self._max_chars:
            cleaned = cleaned[: self._max_chars].rsplit(" ", 1)[0] + "..."

        allowed_names = [name.casefold() for name in allowed_place_names]
        if response_mode == "no_match" and self._mentions_explicit_place(cleaned):
            return self.fallback("place_mentioned_without_matches", response_mode)
        if allowed_names and self._mentions_explicit_place(cleaned):
            lowered = cleaned.casefold()
            has_allowed_name = any(name in lowered for name in allowed_names)
            if not has_allowed_name:
                return self.fallback("mentions_unverified_place", response_mode)

        return GuardedLLMMessage(message=cleaned, used_fallback=False)

    def fallback(
        self,
        reason: str | None = None,
        response_mode: str = "confident",
    ) -> GuardedLLMMessage:
        message = {
            "no_match": NO_MATCH_PLACE_CHAT_FALLBACK,
            "low_confidence": LOW_CONFIDENCE_PLACE_CHAT_FALLBACK,
        }.get(response_mode, DEFAULT_PLACE_CHAT_FALLBACK)
        return GuardedLLMMessage(
            message=message,
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
    def _contains_meta_or_non_plain_response(raw_message: str, cleaned: str) -> bool:
        stripped = raw_message.strip()
        lowered = cleaned.casefold()
        if not stripped:
            return False

        if stripped.startswith(("```", "{", "[", "#")):
            return True
        if re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", raw_message):
            return True
        if (
            len(cleaned) >= 2
            and cleaned[0] in {'"', "'", "“", "«"}
            and cleaned[-1] in {'"', "'", "”", "»"}
        ):
            return True

        meta_patterns = (
            r"\b(?:aqui|aquí)\s+(?:te\s+dejo|tienes|van|hay)\b",
            r"\b(?:dos|2)\s+(?:opciones|alternativas|versiones)\b",
            r"\b(?:opcion|opción|alternativa|version|versión)\s*(?:\d+|uno|dos)?\s*:",
            r"\b(?:ambas|las\s+dos)\s+opciones\b",
            r"\b(?:redactar|redaccion|redacción)\b",
            r"\b(?:busca|buscan)\s+transmitir\b",
        )
        return any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in meta_patterns
        )

    @staticmethod
    def _mentions_explicit_place(message: str) -> bool:
        return bool(
            re.search(
                r"\b(lugar|opci[oó]n|restaurante|caf[eé]|mirador)\s+[A-ZÁÉÍÓÚÑ]",
                message,
            )
        )
