import json
from typing import Any, Sequence

from app.shared.nlp.llm.base import PlaceResponseMode


SYSTEM_PROMPT = """
Eres un redactor conversacional para una app de planes y recomendaciones de lugares.
Tu trabajo es embellecer la respuesta final usando solo los lugares proporcionados por el sistema.

Reglas obligatorias:
- No decidas que lugares recomendar; la lista ya fue seleccionada por FastText, PGVector y filtros.
- No inventes nombres de lugares.
- No inventes horarios, precios, direcciones, calificaciones ni promociones.
- No digas que un lugar esta abierto si el contexto no lo indica.
- Si falta informacion, redacta de forma general.
- Si el usuario pide algo fuera de recomendaciones de lugares o salidas, redirige amablemente al tema de la app.
- No des consejos medicos, legales, financieros o de seguridad.
- La app mostrara las cards desde datos estructurados; no conviertas la respuesta en una tabla.
""".strip()


def build_place_chat_messages(
    user_intent: str,
    region: str | None,
    places: Sequence[dict[str, Any]],
    response_mode: PlaceResponseMode = "confident",
) -> list[dict[str, str]]:
    mode_instruction = {
        "no_match": (
            "Explica con calidez que por ahora no hay lugares que se acoplen a sus "
            "necesidades e invitalo a reformular su plan. No menciones lugares concretos."
        ),
        "low_confidence": (
            "Aclara con tacto que quiza las opciones no sean exactamente lo que busca, "
            "pero que podrian interesarle. No presentes la coincidencia como segura."
        ),
        "confident": (
            "Presenta las opciones de forma positiva porque existe una coincidencia clara."
        ),
    }[response_mode]
    payload = {
        "user_intent": user_intent,
        "region": region,
        "response_mode": response_mode,
        "candidate_places": list(places),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Redacta una respuesta breve, amable y util en espanol. "
                f"Instruccion de tono: {mode_instruction} Contexto: "
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
