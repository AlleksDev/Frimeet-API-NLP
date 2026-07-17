"""Load an open-vocabulary place concept catalog from data, not code rules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.modules.places.infrastructure.open_vocabulary_category_classifier import (
    PlaceCategoryConcept,
)
from app.modules.places.infrastructure.place_semantic_document import PlaceTag
from app.shared.nlp.preprocessing.text import prepare_for_embedding


def load_place_category_concepts(
    catalog_path: str | None,
    *,
    fallback_tags: Iterable[PlaceTag] = (),
) -> tuple[PlaceCategoryConcept, ...]:
    """Load concepts exported by the source system, or derive them from tag data.

    The external catalog format is ``{"concepts": [...]}`` (a bare list is also
    accepted).  It can be refreshed independently of an application deployment.
    The bundled tag inventory is only a backwards-compatible data fallback; no
    category names, aliases or linguistic rules live in this module.
    """

    if catalog_path:
        path = Path(catalog_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Places category catalog not found at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("concepts") if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise ValueError("Places category catalog must contain a concepts list")
        return tuple(_concept_from_record(record) for record in records)

    return _concepts_from_tags(fallback_tags)


def _concepts_from_tags(tags: Iterable[PlaceTag]) -> tuple[PlaceCategoryConcept, ...]:
    concepts: list[PlaceCategoryConcept] = []
    seen: set[str] = set()
    for tag in tags:
        if prepare_for_embedding(tag.category).replace(" ", "_") != "place_category":
            continue
        normalized = prepare_for_embedding(tag.name)
        concept_id = normalized.replace(" ", "_")
        if not concept_id or concept_id in seen:
            continue
        seen.add(concept_id)
        concepts.append(
            PlaceCategoryConcept(
                id=concept_id,
                label=tag.name,
                description=f"Tipo de lugar registrado: {tag.name}",
                storage_values=tuple(dict.fromkeys((tag.name, normalized, concept_id))),
            )
        )
    return tuple(concepts)


def _concept_from_record(record: Any) -> PlaceCategoryConcept:
    if not isinstance(record, Mapping):
        raise ValueError("Each Places category concept must be an object")
    try:
        return PlaceCategoryConcept(
            id=str(record["id"]),
            label=str(record["label"]),
            description=str(record["description"]),
            examples=_text_tuple(record.get("examples")),
            storage_values=_text_tuple(record.get("storage_values")),
        )
    except KeyError as exc:
        raise ValueError(f"Missing Places category concept field: {exc.args[0]}") from exc


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError("Concept examples and storage_values must be lists")
    return tuple(str(item) for item in value)
