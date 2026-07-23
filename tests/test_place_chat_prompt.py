import json

from app.shared.nlp.llm.output_guard import PlaceChatOutputGuard
from app.shared.nlp.prompts.place_chat import build_place_chat_messages


def test_place_chat_prompt_requires_evidence_linked_explanations() -> None:
    messages = build_place_chat_messages(
        user_intent="quiero comer baguettes",
        region="Tuxtla Gutierrez",
        places=[
            {
                "id": "place-1",
                "name": "Pan Local",
                "matched_reasons": ["baguettes", "panaderia"],
                "menu_items": ["Baguette artesanal"],
            }
        ],
    )

    system = messages[0]["content"]
    payload_text = messages[1]["content"].split("Contexto: ", 1)[1]
    payload = json.loads(payload_text)

    assert "evidencia" in system.casefold()
    assert "enlaza" in system.casefold()
    assert "no nombres ni enumeres lugares" in system.casefold()
    assert "tono calido" in system.casefold()
    assert "revisa las cards" in system.casefold()
    assert "exclusivamente un unico mensaje final" in system.casefold()
    assert "solo en texto plano" in system.casefold()
    assert "nunca ofrezcas opciones" in system.casefold()
    assert "un unico mensaje final" in messages[1]["content"].casefold()
    assert payload["candidate_places"][0]["matched_reasons"] == [
        "baguettes",
        "panaderia",
    ]


def test_output_guard_rejects_multiple_draft_options_and_meta_commentary() -> None:
    guarded = PlaceChatOutputGuard().validate(
        message=(
            "Aquí te dejo dos opciones para redactar frases breves y cercanas: "
            '1. "Entiendo que buscas un lugar para nadar." '
            '2. "Quizá estos lugares podrían interesarte." '
            "Ambas opciones buscan transmitir cercanía."
        ),
        allowed_place_names=[],
        response_mode="low_confidence",
    )

    assert guarded.used_fallback is True
    assert guarded.reason == "meta_or_non_plain_response"
    assert "dos opciones" not in guarded.message.casefold()


def test_output_guard_accepts_one_plain_user_facing_message() -> None:
    guarded = PlaceChatOutputGuard().validate(
        message=(
            "Entiendo que quieres encontrar un lugar para nadar y refrescarte. "
            "Todavía no tengo una coincidencia suficientemente clara, pero podemos "
            "probar con otra zona."
        ),
        allowed_place_names=[],
        response_mode="low_confidence",
    )

    assert guarded.used_fallback is False
    assert guarded.reason is None
