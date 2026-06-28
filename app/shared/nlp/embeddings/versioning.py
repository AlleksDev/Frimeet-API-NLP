from app.shared.content_hash import stable_content_hash


def versioned_embedding_hash(
    source_content_hash: str,
    model: str,
    version: str,
    dimension: int,
) -> str:
    """Invalidate derived vectors when content or embedding configuration changes."""
    return stable_content_hash(
        {
            "source_content_hash": source_content_hash,
            "embedding_model": model,
            "embedding_version": version,
            "embedding_dimension": dimension,
        }
    )
