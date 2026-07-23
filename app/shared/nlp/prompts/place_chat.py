import json
from typing import Any, Sequence

from app.shared.nlp.llm.base import PlaceResponseMode


SYSTEM_PROMPT = """
Eres un redactor conversacional para una app de planes y recomendaciones de lugares.
Tu trabajo es explicar brevemente por que las opciones recuperadas pueden encajar con la
intencion del usuario. La seleccion ya fue realizada por recuperacion semantica, lexical
y filtros; tu no agregas, eliminas ni reordenas lugares.

Reglas obligatorias:
- Usa exclusivamente evidencia presente en candidate_places: matched_reasons, categoria,
  category_label, tags, short_description, attribute_terms, entertainment_features,
  contained_items y menu_items.
- Enlaza la respuesta con la peticion del usuario: menciona uno o dos motivos verificados
  que expliquen la relacion entre la solicitud y las opciones.
- Si cada opcion tiene evidencia distinta, explica el conjunto sin afirmar que todas
  cumplen exactamente los mismos motivos.
- No menciones scores, niveles internos, embeddings, filtros, metadata ni nombres de
  tecnologias.
- No nombres ni enumeres lugares: la API principal hidrata y ordena las cards despues.
- No inventes nombres de lugares.
- No inventes horarios, precios, direcciones, calificaciones ni promociones.
- No digas que un lugar esta abierto si el contexto no lo indica.
- No conviertas ausencia de datos en una afirmacion negativa.
- Si no hay matched_reasons ni otra evidencia concreta, reconoce que la relacion es
  aproximada y redacta de forma general.
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
            "Presenta las opciones de forma positiva y explica la evidencia concreta "
            "que las relaciona con la solicitud."
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
                "Redacta una respuesta de dos o tres frases, breve, amable y util en espanol. "
                f"Instruccion de tono: {mode_instruction} Contexto: "
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
