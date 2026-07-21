"""Compatibility policy for persisted Places chat taxonomy state."""

from __future__ import annotations


_COMPATIBLE_PREVIOUS_VERSIONS: dict[str, frozenset[str]] = {
    # V2 only adds category storage aliases and searchable preferences. Existing
    # V1 canonical IDs and clarification option IDs keep the same meaning, so a
    # turn can safely consume V1 state and re-emit a V2 state patch.
    "places-taxonomy-v2": frozenset({"places-taxonomy-v1"}),
}


def is_compatible_place_chat_taxonomy(
    persisted_version: str | None,
    current_version: str,
) -> bool:
    if persisted_version is None or persisted_version == current_version:
        return True
    return persisted_version in _COMPATIBLE_PREVIOUS_VERSIONS.get(
        current_version,
        frozenset(),
    )
