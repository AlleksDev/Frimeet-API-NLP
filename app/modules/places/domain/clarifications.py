from collections.abc import Mapping, Sequence
import re
import unicodedata
from uuid import uuid4

from app.modules.places.domain.chat_intent import (
    Clarification,
    ClarificationKind,
    ClarificationOption,
    PendingClarification,
    PendingClarificationOption,
    ResolvedPlaceAnchor,
)


_SPANISH_CHAT_CATEGORY_LABELS = {
    "cafe": "Cafeterías",
    "restaurant": "Restaurantes",
    "park": "Parques",
    "bar": "Bares",
    "nightlife": "Vida nocturna",
    "culture": "Cultura y museos",
    "shopping": "Compras",
    "sports": "Deportes",
    "bakery": "Panaderías",
    "ice_cream": "Heladerías y postres",
    "cinema": "Cines",
    "library": "Bibliotecas",
    "market": "Mercados",
    "outdoors": "Actividades al aire libre",
    "lodging": "Alojamiento",
}


def new_category_clarification(
    categories: Sequence[str],
    *,
    kind: ClarificationKind = "target_category",
    labels: Mapping[str, str] | None = None,
) -> PendingClarification:
    unique = tuple(dict.fromkeys(category for category in categories if category))[:5]
    if len(unique) < 2:
        raise ValueError("a category clarification requires at least two options")
    option_labels = labels or {}
    option_ids = _category_option_ids(unique)
    return PendingClarification(
        clarification_id=str(uuid4()),
        kind=kind,
        options=tuple(
            PendingClarificationOption(
                option_id=option_id,
                value=category,
                label=_bounded_text(
                    category_display_label(
                        category,
                        option_labels.get(category),
                    ),
                    160,
                ),
            )
            for category, option_id in zip(unique, option_ids)
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


def category_display_label(category: str, label: str | None = None) -> str:
    normalized = category.strip().casefold()
    value = " ".join(
        (
            label
            or _SPANISH_CHAT_CATEGORY_LABELS.get(normalized)
            or category
        ).replace("_", " ").split()
    ).strip()
    if not value:
        value = "Lugar"
    return value[:1].upper() + value[1:]


def _category_option_ids(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    option_ids: list[str] = []
    for index, value in enumerate(values, start=1):
        option_id = _category_option_id(value, index)
        if option_id in seen:
            suffix = f"_{index}"
            option_id = option_id[: 64 - len(suffix)].rstrip("_-") + suffix
        seen.add(option_id)
        option_ids.append(option_id)
    return tuple(option_ids)


def _category_option_id(value: str, index: int) -> str:
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        return value
    ascii_value = unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", ascii_value).strip("_-").lower()
    slug = slug[:54].rstrip("_-") or "category"
    return f"{slug}_{index}"
