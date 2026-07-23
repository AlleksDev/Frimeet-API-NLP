import re
from typing import Any

from app.modules.places.domain.models import PlaceCandidate
from app.shared.nlp.preprocessing.text import prepare_for_embedding


NAME_WEIGHT = 5
DESCRIPTION_WEIGHT = 3
CATEGORY_WEIGHT = 2
MENU_WEIGHT = 3
TAG_WEIGHT = 1
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "al",
    "algo",
    "algun",
    "alguna",
    "algunas",
    "algunos",
    "con",
    "cual",
    "cuando",
    "de",
    "del",
    "donde",
    "el",
    "ella",
    "en",
    "es",
    "esta",
    "este",
    "hay",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "para",
    "pero",
    "por",
    "que",
    "quiero",
    "se",
    "ser",
    "su",
    "sus",
    "te",
    "tener",
    "tu",
    "tus",
    "un",
    "una",
    "unas",
    "uno",
    "unos",
    "ver",
    "y",
    "ya",
    "yo",
    "busco",
    "buscar",
    "lugar",
    "lugares",
    "necesito",
    "puedo",
}


def place_tokens(place: PlaceCandidate) -> list[str]:
    tags = _as_text(place.metadata.get("tags"))
    category = place.category or ""
    description = _as_text(place.metadata.get("short_description"))
    menu = _as_text(place.metadata.get("menu_items"))
    facet_document = " ".join(
        value
        for value in (
            _as_text(place.metadata.get("attribute_terms")),
            _as_text(place.metadata.get("entertainment_features")),
            _as_text(place.metadata.get("contained_items")),
            menu,
        )
        if value
    )
    base_document = place.document or " ".join(
        value
        for value in [
            place.name,
            category,
            place.city or "",
            place.state or "",
            tags,
            _as_text(place.metadata.get("occasion")),
            description,
        ]
        if value
    )

    # Metadata facets are appended even when a stored semantic document exists.
    # This keeps lexical matching correct during rolling re-indexes from v3 to
    # v4, while the content hash still guarantees eventual document refresh.
    weighted_fields = [base_document, facet_document]
    weighted_fields.extend([place.name] * (NAME_WEIGHT - 1))
    weighted_fields.extend([description] * (DESCRIPTION_WEIGHT - 1))
    weighted_fields.extend([category] * (CATEGORY_WEIGHT - 1))
    weighted_fields.extend([menu] * (MENU_WEIGHT - 1))
    # Tags remain searchable once through base_document. Repeating them used
    # to let broad or noisy labels dominate the actual name and description.
    return tokenize(" ".join(value for value in weighted_fields if value))


def tokenize(text: str) -> list[str]:
    normalized = prepare_for_embedding(text)
    return [
        stem_spanish_token(token)
        for token in TOKEN_PATTERN.findall(normalized)
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def stem_spanish_token(token: str) -> str:
    """Apply a small deterministic stemmer for common Spanish variants."""
    if token.isdigit() or len(token) <= 3:
        return token

    stem = token
    if len(stem) > 5 and stem.endswith("es"):
        stem = stem[:-2]
    elif len(stem) > 4 and stem.endswith("s"):
        stem = stem[:-1]

    for suffix in ("ando", "iendo", "ados", "adas", "idos", "idas"):
        if len(stem) > len(suffix) + 3 and stem.endswith(suffix):
            return stem[: -len(suffix)]

    for suffix in ("ar", "er", "ir"):
        if len(stem) > len(suffix) + 2 and stem.endswith(suffix):
            return stem[: -len(suffix)]

    if len(stem) > 3 and stem.endswith(("a", "o")):
        stem = stem[:-1]
    return stem


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)
