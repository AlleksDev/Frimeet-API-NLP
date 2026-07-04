from typing import Any

from app.shared.content_hash import stable_content_hash
from app.shared.search.source import SearchSourceRecord


def first_present(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [], (), {})
    }


def make_search_record(
    *,
    resource_id: Any,
    document: str,
    metadata: dict[str, Any],
    document_version: str,
    is_active: bool = True,
) -> SearchSourceRecord:
    filtered_metadata = compact_metadata(metadata)
    content_hash = stable_content_hash(
        {
            "document": document,
            "metadata": filtered_metadata,
            "is_active": is_active,
            "document_version": document_version,
        }
    )
    return SearchSourceRecord(
        id=str(resource_id),
        document=document,
        metadata=filtered_metadata,
        content_hash=content_hash,
        is_active=is_active,
    )
