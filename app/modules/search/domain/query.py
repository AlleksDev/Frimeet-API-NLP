import re
import unicodedata


def normalize_search_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", (value or "").casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()
