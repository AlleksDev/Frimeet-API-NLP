import re
from typing import Any

from app.modules.places.domain.models import PlaceCandidate
from app.shared.nlp.preprocessing.text import prepare_for_embedding


TAG_WEIGHT = 6
CATEGORY_WEIGHT = 2
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
    base_document = place.document or " ".join(
        value
        for value in [
            place.name,
            category,
            place.city or "",
            place.state or "",
            tags,
            _as_text(place.metadata.get("occasion")),
            _as_text(place.metadata.get("short_description")),
        ]
        if value
    )

    weighted_fields = [base_document]
    weighted_fields.extend([category] * (CATEGORY_WEIGHT - 1))
    weighted_fields.extend([tags] * (TAG_WEIGHT - 1))
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
