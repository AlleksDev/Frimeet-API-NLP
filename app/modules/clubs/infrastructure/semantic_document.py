from app.shared.nlp.embeddings.weighted_document import build_weighted_document

CLUB_SEARCH_DOCUMENT_VERSION = "club-search-v1"
CLUB_SEARCH_FIELD_WEIGHTS = {"name": 8, "category": 5, "description": 3}


def build_club_search_document(name: str, category: str, description: str) -> str:
    return build_weighted_document(
        [
            (name, CLUB_SEARCH_FIELD_WEIGHTS["name"]),
            (category, CLUB_SEARCH_FIELD_WEIGHTS["category"]),
            (description, CLUB_SEARCH_FIELD_WEIGHTS["description"]),
        ]
    )
