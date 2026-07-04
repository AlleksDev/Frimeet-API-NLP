from app.shared.nlp.embeddings.weighted_document import build_weighted_document

USER_SEARCH_DOCUMENT_VERSION = "user-search-v1"
USER_SEARCH_FIELD_WEIGHTS = {"username": 10, "full_name": 8, "bio": 2}


def build_user_search_document(username: str, full_name: str, bio: str) -> str:
    return build_weighted_document(
        [
            (username, USER_SEARCH_FIELD_WEIGHTS["username"]),
            (full_name, USER_SEARCH_FIELD_WEIGHTS["full_name"]),
            (bio, USER_SEARCH_FIELD_WEIGHTS["bio"]),
        ]
    )
