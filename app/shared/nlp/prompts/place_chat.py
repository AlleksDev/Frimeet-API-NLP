import json
from typing import Any, Sequence


SYSTEM_PROMPT = """
Eres un redactor conversacional para una app de planes y recomendaciones de lugares.
Tu trabajo es embellecer la respuesta final usando solo los lugares proporcionados por el sistema.

Reglas obligatorias:
- No decidas que lugares recomendar; la lista ya fue seleccionada por embeddings, filtros y ranking.
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
) -> list[dict[str, str]]:
    payload = {
        "user_intent": user_intent,
        "region": region,
        "candidate_places": list(places),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Redacta una respuesta breve, amable y util en espanol con este contexto: "
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
