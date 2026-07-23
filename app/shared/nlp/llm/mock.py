from typing import Any, Sequence

from app.shared.nlp.llm.base import LLMProvider, LLMResult, PlaceResponseMode


class MockLLMProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-llama") -> None:
        self.model_name = model_name

    async def generate_place_chat_response(
        self,
        user_intent: str,
        region: str | None,
        places: Sequence[dict[str, Any]],
        response_mode: PlaceResponseMode = "confident",
    ) -> LLMResult:
        del user_intent, region
        names = [str(place["name"]) for place in places if place.get("name")]
        if response_mode == "no_match":
            message = (
                "Por ahora no encontré lugares que se acoplen bien a tus necesidades. "
                "Prueba contándome otro tipo de plan, zona u ocasión y lo intentamos de nuevo."
            )
        elif response_mode == "low_confidence" and names:
            message = (
                "Encontré algunas posibilidades cercanas a tu idea: "
                + ", ".join(names[:3])
                + ". Todavía no tengo señales suficientes para asegurarte que sean "
                "justo lo que buscas; cuéntame qué detalle no puede faltar y lo "
                "afinamos juntos."
            )
        elif names:
            message = (
                "¡Claro! Encontré algunas opciones que se acercan bastante a tu plan: "
                + ", ".join(names[:3])
                + ". Ojalá alguna se convierta en una buena salida para ti."
            )
        else:
            message = (
                "No encontré lugares suficientemente cercanos a tu búsqueda. "
                "Puedes intentar con otra zona, ocasión o tipo de plan."
            )
        return LLMResult(
            message=message,
            provider=self.provider_name,
            model=self.model_name,
        )
