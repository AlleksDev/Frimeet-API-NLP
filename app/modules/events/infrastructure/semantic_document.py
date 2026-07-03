from app.shared.nlp.embeddings.weighted_document import build_weighted_document

EVENT_SEARCH_DOCUMENT_VERSION = "event-search-v1"
EVENT_SEARCH_FIELD_WEIGHTS = {"title": 8, "tags": 5, "description": 3}


def build_event_search_document(
    title: str,
    tag_names: list[str],
    description: str,
) -> str:
    return build_weighted_document(
        [
            (title, EVENT_SEARCH_FIELD_WEIGHTS["title"]),
            (" ".join(tag_names), EVENT_SEARCH_FIELD_WEIGHTS["tags"]),
            (description, EVENT_SEARCH_FIELD_WEIGHTS["description"]),
        ]
    )
