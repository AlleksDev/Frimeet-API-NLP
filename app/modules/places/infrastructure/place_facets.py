"""Normalize the searchable place facets exposed by the main API.

The source schema distinguishes unknown values (``NULL``) from explicit
absence (``false``/``no``).  Only confirmed positive facets are embedded so a
place that explicitly has no parking cannot match a positive parking request.
The explicit negative state is still preserved in metadata for diagnostics and
future structured filters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.shared.nlp.preprocessing.text import clean_text


_BOOLEAN_LABELS: dict[str, str] = {
    "has_kids_playroom": "area de juegos para ninas y ninos",
    "has_restrooms": "banos y sanitarios",
    "has_parking": "estacionamiento",
    "has_entertainment": "entretenimiento",
    "has_romantic_space": "espacio romantico para parejas",
    "has_family_space": "espacio familiar",
    "has_friends_space": "espacio para amigos",
    "has_screens": "pantallas",
    "has_seating": "asientos y lugares para sentarse",
    "allows_outside_food": "permite ingresar comida del exterior",
    "allows_backpack": "permite ingresar con mochila",
}

_PRESENCE_LABELS: dict[str, dict[str, str]] = {
    "kid_friendly": {
        "yes": "apto para ninas, ninos y familias",
        "normal": "ambiente familiar moderado",
    },
    "good_for_cooling_off": {
        "yes": (
            "apto para refrescarse, nadar o realizar actividades acuaticas"
        ),
        "normal": "opcion moderada para refrescarse",
    },
}

_PRICE_TIER_LABELS = {
    "cheap": "precio barato",
    "accessible": "precio accesible",
    "normal": "precio medio",
    "expensive": "precio alto",
    "very_expensive": "precio muy alto",
    "deluxe": "precio de lujo",
}


@dataclass(frozen=True)
class ResolvedPlaceFacets:
    attribute_states: dict[str, bool | str]
    positive_attributes: tuple[str, ...]
    negative_attributes: tuple[str, ...]
    entertainment_features: tuple[str, ...]
    contained_items: tuple[str, ...]
    menu_items: tuple[str, ...]

    @property
    def positive_evidence(self) -> tuple[str, ...]:
        return _ordered_unique(
            (
                *self.positive_attributes,
                *self.entertainment_features,
                *self.contained_items,
                *self.menu_items,
            )
        )


def resolve_place_facets(
    attributes: Any,
    entertainment: Any,
    contained_items: Any = None,
    menu: Any = None,
) -> ResolvedPlaceFacets:
    states: dict[str, bool | str] = {}
    positive: list[str] = []
    negative: list[str] = []
    source_attributes = attributes if isinstance(attributes, Mapping) else {}

    for key, label in _BOOLEAN_LABELS.items():
        value = _optional_bool(source_attributes.get(key))
        if value is None:
            continue
        states[key] = value
        (positive if value else negative).append(label)

    for key, labels in _PRESENCE_LABELS.items():
        value = _normalized_enum(source_attributes.get(key))
        if value not in {"yes", "no", "normal"}:
            continue
        states[key] = value
        if value == "no":
            negative.append(_humanize_key(key))
        else:
            positive.append(labels[value])

    price_tier = _normalized_enum(source_attributes.get("price_tier"))
    if price_tier in _PRICE_TIER_LABELS:
        states["price_tier"] = price_tier
        positive.append(_PRICE_TIER_LABELS[price_tier])

    entertainment_note = clean_text(
        str(source_attributes.get("entertainment_note") or "")
    )
    if entertainment_note:
        states["entertainment_note"] = entertainment_note
        if states.get("has_entertainment") is not False:
            positive.append(entertainment_note)

    entertainment_features = _record_terms(
        entertainment,
        fields=("label", "feature_key"),
        maximum=40,
    )
    if states.get("has_entertainment") is False:
        entertainment_features = ()

    return ResolvedPlaceFacets(
        attribute_states=states,
        positive_attributes=_ordered_unique(positive),
        negative_attributes=_ordered_unique(negative),
        entertainment_features=entertainment_features,
        contained_items=_record_terms(
            contained_items,
            fields=("name", "item_type", "category", "description"),
            maximum=80,
        ),
        menu_items=_menu_terms(menu),
    )


def _record_terms(
    value: Any,
    *,
    fields: tuple[str, ...],
    maximum: int,
) -> tuple[str, ...]:
    records = _as_records(value)
    terms: list[str] = []
    for record in records[:maximum]:
        parts: list[str] = []
        for field in fields:
            raw = record.get(field)
            if raw is None:
                continue
            text = clean_text(str(raw).replace("_", " "))
            if text and text.casefold() not in {part.casefold() for part in parts}:
                parts.append(text)
        if parts:
            terms.append(" ".join(parts))
    return _ordered_unique(terms)


def _menu_terms(value: Any) -> tuple[str, ...]:
    """Index only menu items that can actually be ordered.

    ``is_available`` is non-null in the source schema, but treating an omitted
    value as available keeps this consumer compatible with older snapshots.
    An explicit ``false`` is authoritative and must not become search evidence.
    """

    available_records = [
        record
        for record in _as_records(value)
        if _optional_bool(record.get("is_available")) is not False
    ]
    return _record_terms(
        available_records,
        fields=("name", "category", "description"),
        maximum=80,
    )


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested = value.get("items")
        if isinstance(nested, Sequence) and not isinstance(nested, str | bytes):
            value = nested
        else:
            value = [value]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "si", "sí"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _normalized_enum(value: Any) -> str:
    return clean_text(str(value or "")).casefold().replace(" ", "_")


def _humanize_key(value: str) -> str:
    return clean_text(value.replace("_", " "))


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)
