from app.shared.content_hash import stable_content_hash


def versioned_embedding_hash(
    source_content_hash: str,
    model: str,
    version: str,
    dimension: int,
    *,
    revision: str | None = None,
    fix_mistral_regex: bool | None = None,
) -> str:
    """Invalidate derived vectors when content or embedding configuration changes."""
    payload: dict[str, str | int | bool] = {
        "source_content_hash": source_content_hash,
        "embedding_model": model,
        "embedding_version": version,
        "embedding_dimension": dimension,
    }
    if revision is not None:
        payload["embedding_model_revision"] = revision
    if fix_mistral_regex is not None:
        payload["embedding_fix_mistral_regex"] = fix_mistral_regex
    return stable_content_hash(payload)
