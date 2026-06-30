from app.shared.content_hash import stable_content_hash


def versioned_embedding_hash(
    source_content_hash: str,
    model: str,
    version: str,
    dimension: int,
    provider: str = "",
    query_prefix: str = "",
    document_prefix: str = "",
) -> str:
    """Invalidate derived vectors when content or embedding configuration changes."""
    return stable_content_hash(
        {
            "source_content_hash": source_content_hash,
            "embedding_model": model,
            "embedding_version": version,
            "embedding_dimension": dimension,
            "embedding_provider": provider,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
        }
    )
