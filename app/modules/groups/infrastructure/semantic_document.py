from app.shared.nlp.embeddings.weighted_document import build_weighted_document

GROUP_SEARCH_DOCUMENT_VERSION = "group-search-v1"
GROUP_SEARCH_FIELD_WEIGHTS = {"name": 8, "description": 3}


def build_group_search_document(name: str, description: str) -> str:
    return build_weighted_document(
        [
            (name, GROUP_SEARCH_FIELD_WEIGHTS["name"]),
            (description, GROUP_SEARCH_FIELD_WEIGHTS["description"]),
        ]
    )
