import json

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
    assert payload["candidate_places"][0]["matched_reasons"] == [
        "baguettes",
        "panaderia",
    ]
