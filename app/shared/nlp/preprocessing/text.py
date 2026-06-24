import re
import unicodedata


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    return remove_extra_spaces(text)


def normalize_text(text: str, remove_accents: bool = True) -> str:
    cleaned = clean_text(text).casefold()
    if remove_accents:
        cleaned = strip_accents(cleaned)
    return remove_extra_spaces(cleaned)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def prepare_for_embedding(text: str) -> str:
    return normalize_text(text, remove_accents=True)
