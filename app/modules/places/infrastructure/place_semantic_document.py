from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from app.shared.nlp.preprocessing.text import clean_text
from app.modules.places.infrastructure.place_facets import ResolvedPlaceFacets


PLACE_SEMANTIC_FIELD_WEIGHTS = {
    "tags": 1,
    "category": 2,
    "description": 3,
    "name": 5,
    "attributes": 1,
    "entertainment": 1,
    "contained_items": 1,
    "menu": 3,
}
PLACE_SEMANTIC_DOCUMENT_VERSION = "structured-place-v4"


@dataclass(frozen=True)
class PlaceTag:
    id: int
    name: str
    category: str


@dataclass(frozen=True)
class ResolvedPlaceTags:
    names: tuple[str, ...]
    ids: tuple[int, ...]
    categories: tuple[str, ...]
    unknown_ids: tuple[int, ...]


def build_place_semantic_document(
    name: str,
    category: Any,
    description: str,
    resolved_tags: ResolvedPlaceTags,
    *,
    category_label: Any = None,
    resolved_facets: ResolvedPlaceFacets | None = None,
) -> str:
    # Transformer encoders use context and sentence structure; repeating tokens
    # to simulate weights (the old FastText strategy) distorts that context.
    # Keep every source value once and expose its role explicitly instead.
    facets = resolved_facets or ResolvedPlaceFacets({}, (), (), (), (), ())
    fields = (
        ("Nombre", clean_text(name)),
        (
            "Categoria",
            _category_semantic_text(category, category_label),
        ),
        ("Descripcion", clean_text(description)),
        ("Etiquetas", " ".join(resolved_tags.names)),
        ("Familias de etiquetas", " ".join(resolved_tags.categories)),
        ("Atributos confirmados", ", ".join(facets.positive_attributes)),
        (
            "Entretenimiento disponible",
            ", ".join(facets.entertainment_features),
        ),
        ("Elementos y actividades", ", ".join(facets.contained_items)),
        ("Menu", ", ".join(facets.menu_items)),
    )
    return " ".join(
        f"{label}: {value}."
        for label, value in fields
        if value
    ).strip()


def semantic_category_text(category: Any) -> str:
    raw_category = clean_text(str(category or "")).casefold().replace("_", " ")
    if not raw_category:
        return ""
    return raw_category


def _category_semantic_text(category: Any, category_label: Any) -> str:
    canonical = semantic_category_text(category)
    localized = clean_text(str(category_label or ""))
    if not localized:
        return canonical
    if localized.casefold() == canonical.casefold():
        return localized
    return " ".join(value for value in (localized, canonical) if value)


def resolve_place_tags(value: Any) -> ResolvedPlaceTags:
    resolved_names: list[str] = []
    tag_ids: list[int] = []
    tag_categories: list[str] = []
    unknown_ids: list[int] = []
    seen_names: set[str] = set()

    for raw_tag in _as_tag_values(value):
        tag_payload = raw_tag if isinstance(raw_tag, Mapping) else None
        tag_id = _as_tag_id(
            tag_payload.get("id") if tag_payload is not None else raw_tag
        )
        if tag_id is not None:
            tag_ids.append(tag_id)
            supplied_name = (
                tag_payload.get("label") or tag_payload.get("name")
                if tag_payload is not None
                else None
            )
            tag = place_tag_catalog().get(tag_id)
            if supplied_name:
                name = str(supplied_name).replace("_", " ")
                supplied_category = tag_payload.get("category")
                if supplied_category:
                    tag_categories.append(str(supplied_category))
                elif tag is not None:
                    tag_categories.append(tag.category)
            elif tag is None:
                unknown_ids.append(tag_id)
                continue
            else:
                name = tag.name.replace("_", " ")
                tag_categories.append(tag.category)
        else:
            supplied_name = (
                tag_payload.get("label") or tag_payload.get("name")
                if tag_payload is not None
                else raw_tag
            )
            name = str(supplied_name or "").replace("_", " ")

        cleaned_name = clean_text(name)
        normalized_name = cleaned_name.casefold()
        if cleaned_name and normalized_name not in seen_names:
            seen_names.add(normalized_name)
            resolved_names.append(cleaned_name)

    return ResolvedPlaceTags(
        names=tuple(resolved_names),
        ids=tuple(dict.fromkeys(tag_ids)),
        categories=tuple(dict.fromkeys(tag_categories)),
        unknown_ids=tuple(dict.fromkeys(unknown_ids)),
    )


@lru_cache
def place_tag_catalog() -> dict[int, PlaceTag]:
    catalog_path = Path(__file__).with_name("place_tag_catalog.json")
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        int(item["id"]): PlaceTag(
            id=int(item["id"]),
            name=str(item["name"]),
            category=str(item["category"]),
        )
        for item in payload["data"]
    }


def _as_tag_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _as_tag_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
