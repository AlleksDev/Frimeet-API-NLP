from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from app.shared.nlp.embeddings.weighted_document import build_weighted_document
from app.shared.nlp.preprocessing.text import clean_text


PLACE_SEMANTIC_FIELD_WEIGHTS = {
    "tags": 6,
    "category": 4,
    "description": 3,
    "name": 1,
}
PLACE_SEMANTIC_DOCUMENT_VERSION = "weighted-tags-v2"


# Broad categories from the main API are expanded into Spanish intent terms so
# sparse OSM records still have a useful semantic anchor.
CATEGORY_SEMANTIC_PROFILES = {
    "restaurant": "restaurante comida gastronomia comer cena almuerzo desayuno",
    "cafe": "cafe cafeteria bebidas desayuno postres conversar",
    "bar": "bar bebidas cocteles cerveza amigos musica noche",
    "nightlife": "vida nocturna noche baile musica bar fiesta",
    "shopping": "compras tiendas ropa calzado productos mercado centro comercial",
    "lodging": "alojamiento hotel hospedaje hostal dormir turismo viaje",
    "park": "parque naturaleza caminar paseo aire libre mascotas ejercicio",
    "culture": "cultura museo arte historia biblioteca exposicion lectura",
    "tourism": "turismo atraccion visitar explorar paseo historia",
    "sports": "deporte ejercicio entrenamiento gimnasio actividad fisica",
    "community": "comunidad convivencia reuniones centro comunitario actividades",
    "family": "familia ninos juegos convivencia actividades familiares",
    "entertainment": "entretenimiento diversion juegos cine actividades",
}


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
) -> str:
    tags_text = " ".join(resolved_tags.names)
    return build_weighted_document(
        [
            (name, PLACE_SEMANTIC_FIELD_WEIGHTS["name"]),
            (
                semantic_category_text(category),
                PLACE_SEMANTIC_FIELD_WEIGHTS["category"],
            ),
            (description, PLACE_SEMANTIC_FIELD_WEIGHTS["description"]),
            (tags_text, PLACE_SEMANTIC_FIELD_WEIGHTS["tags"]),
        ]
    )


def semantic_category_text(category: Any) -> str:
    raw_category = clean_text(str(category or "")).casefold().replace("_", " ")
    if not raw_category:
        return ""
    profile_key = raw_category.replace(" ", "_")
    return CATEGORY_SEMANTIC_PROFILES.get(profile_key, raw_category)


def resolve_place_tags(value: Any) -> ResolvedPlaceTags:
    resolved_names: list[str] = []
    tag_ids: list[int] = []
    tag_categories: list[str] = []
    unknown_ids: list[int] = []
    seen_names: set[str] = set()

    for raw_tag in _as_tag_values(value):
        tag_id = _as_tag_id(raw_tag)
        if tag_id is not None:
            tag_ids.append(tag_id)
            tag = place_tag_catalog().get(tag_id)
            if tag is None:
                unknown_ids.append(tag_id)
                continue
            name = tag.name.replace("_", " ")
            tag_categories.append(tag.category)
        else:
            name = str(raw_tag).replace("_", " ")

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
