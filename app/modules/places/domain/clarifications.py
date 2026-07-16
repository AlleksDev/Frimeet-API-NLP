from collections.abc import Sequence
from uuid import uuid4

from app.modules.places.domain.chat_intent import (
    Clarification,
    ClarificationKind,
    ClarificationOption,
    PendingClarification,
    PendingClarificationOption,
    ResolvedPlaceAnchor,
)


_CATEGORY_LABELS = {
    "restaurant": "Restaurantes",
    "cafe": "Cafeterias",
    "park": "Parques",
    "nightlife": "Fiesta y vida nocturna",
    "sports": "Ejercicio y deporte",
    "cinema": "Cines",
    "shopping": "Compras",
    "lodging": "Hospedaje",
}


def new_category_clarification(
    categories: Sequence[str],
    *,
    kind: ClarificationKind = "target_category",
) -> PendingClarification:
    unique = tuple(dict.fromkeys(category for category in categories if category))[:5]
    if len(unique) < 2:
        raise ValueError("a category clarification requires at least two options")
    return PendingClarification(
        clarification_id=str(uuid4()),
        kind=kind,
        options=tuple(
            PendingClarificationOption(
                option_id=category,
                value=category,
                label=_CATEGORY_LABELS.get(category, category.replace("_", " ").title()),
            )
            for category in unique
        ),
    )


def new_location_scope_clarification(
    anchor_text: str,
    radius_meters: int | None = None,
) -> PendingClarification:
    label = anchor_text.title()
    return PendingClarification(
        clarification_id=str(uuid4()),
        kind="location_scope",
        location_anchor_text=anchor_text,
        radius_meters=radius_meters,
        strict_radius=radius_meters is not None,
        options=(
            PendingClarificationOption(
                option_id="target_results",
                value="target_results",
                label=f"Buscar cerca de {label}",
            ),
            PendingClarificationOption(
                option_id="reference_entity",
                value="reference_entity",
                label="Usarlo solo para identificar la referencia",
            ),
        ),
    )


def new_anchor_clarification(
    kind: ClarificationKind,
    anchor_text: str,
    anchors: Sequence[ResolvedPlaceAnchor],
) -> PendingClarification:
    bounded = tuple(anchors[:5])
    if len(bounded) < 2:
        raise ValueError("an anchor clarification requires at least two options")
    return PendingClarification(
        clarification_id=str(uuid4()),
        kind=kind,
        location_anchor_text=anchor_text,
        options=tuple(
            PendingClarificationOption(
                option_id=f"option_{index}",
                value=anchor.place_id,
                label=anchor.name,
                place_id=anchor.place_id,
                attributes=anchor.attributes,
            )
            for index, anchor in enumerate(bounded, start=1)
        ),
    )


def ensure_legacy_pending_options(
    pending: PendingClarification,
) -> PendingClarification:
    if len(pending.options) >= 2:
        return pending
    if pending.kind == "location_scope" and pending.location_anchor_text:
        refreshed = new_location_scope_clarification(
            pending.location_anchor_text,
            pending.radius_meters,
        )
        return PendingClarification(
            clarification_id=pending.clarification_id or refreshed.clarification_id,
            kind=refreshed.kind,
            options=refreshed.options,
            location_anchor_text=refreshed.location_anchor_text,
            radius_meters=refreshed.radius_meters,
            strict_radius=refreshed.strict_radius,
        )
    raise ValueError("pending clarification has no actionable options")


def to_public_clarification(pending: PendingClarification) -> Clarification:
    if not 2 <= len(pending.options) <= 5:
        raise ValueError("clarification must contain between two and five options")
    if len({option.option_id for option in pending.options}) != len(pending.options):
        raise ValueError("clarification option ids must be unique")
    return Clarification(
        clarification_id=pending.clarification_id,
        kind=pending.kind,
        prompt=_prompt(pending),
        options=tuple(
            ClarificationOption(
                option_id=option.option_id,
                label=_bounded_text(option.label, 80),
                message=_bounded_text(option.label, 200),
            )
            for option in pending.options
        ),
    )


def _prompt(pending: PendingClarification) -> str:
    anchor = (pending.location_anchor_text or "ese lugar").title()
    return {
        "target_category": "¿Que tipo de lugar prefieres?",
        "intent_category": "¿Cual de estas opciones se acerca mas a lo que buscas?",
        "location_scope": f"¿Como quieres usar {anchor}?",
        "location_anchor": f"Encontre varios lugares para {anchor}. ¿Cual es?",
        "reference_entity": "Encontre varias referencias posibles. ¿Cual quieres usar?",
        "reference_location_anchor": (
            f"Encontre varias ubicaciones para {anchor}. ¿Cual es?"
        ),
    }[pending.kind]


def _bounded_text(value: str, maximum: int) -> str:
    value = " ".join(value.split()).strip()
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"
