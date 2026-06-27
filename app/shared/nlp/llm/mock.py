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
                "Quizá no sean exactamente lo que estás buscando, pero podrían interesarte: "
                + ", ".join(names[:3])
                + ". Échales un vistazo y decide si alguna encaja con tu plan."
            )
        elif names:
            message = (
                "Encontré algunas opciones reales que pueden encajar con tu plan: "
                + ", ".join(names[:3])
                + ". Revisa sus detalles y elige la que mejor vaya con la salida."
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
