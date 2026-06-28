import re
import unicodedata


SEMANTIC_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SEMANTIC_STOPWORDS = {
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


def tokenize_for_embeddings(text: str) -> list[str]:
    """Tokenize Spanish text for mean FastText document embeddings."""
    normalized = prepare_for_embedding(text)
    tokens = [
        token
        for token in SEMANTIC_TOKEN_PATTERN.findall(normalized)
        if token not in SEMANTIC_STOPWORDS
        and (len(token) > 1 or token.isdigit())
    ]
    if tokens:
        return tokens
    return SEMANTIC_TOKEN_PATTERN.findall(normalized)
