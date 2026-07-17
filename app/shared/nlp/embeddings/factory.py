from app.shared.config.settings import Settings
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.fasttext import FastTextEmbeddingProvider
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.casefold()
    if provider == "fasttext":
        return FastTextEmbeddingProvider(
            model_path=settings.fasttext_model_path,
            expected_dimension=settings.embedding_dimension,
            repo_id=settings.fasttext_model_repo_id,
            filename=settings.fasttext_model_filename,
            auto_download=settings.fasttext_auto_download,
        )
    if provider == "mock":
        return MockEmbeddingProvider(dimension=settings.embedding_dimension)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def create_place_embedding_provider(
    settings: Settings,
    *,
    text_role: str = "query",
) -> EmbeddingProvider:
    """Create the Places-only embedding provider.

    Keeping this separate from ``create_embedding_provider`` prevents a Places
    migration from changing post, feed or global-search vector dimensions.
    ``text_role`` selects the query/passage prefix required by retrieval models
    such as E5 while sharing the same model weights.
    """

    if text_role not in {"query", "passage"}:
        raise ValueError("text_role must be 'query' or 'passage'")

    provider = settings.places_embedding_provider.casefold()
    if provider == "fasttext":
        return FastTextEmbeddingProvider(
            model_path=settings.fasttext_model_path,
            expected_dimension=settings.places_embedding_dimension,
            repo_id=settings.fasttext_model_repo_id,
            filename=settings.fasttext_model_filename,
            auto_download=settings.fasttext_auto_download,
        )
    if provider == "mock":
        return MockEmbeddingProvider(dimension=settings.places_embedding_dimension)
    if provider in {"sentence_transformer", "bert"}:
        prefix = (
            settings.places_embedding_query_prefix
            if text_role == "query"
            else settings.places_embedding_passage_prefix
        )
        return SentenceTransformerEmbeddingProvider(
            model_name_or_path=settings.places_embedding_model,
            expected_dimension=settings.places_embedding_dimension,
            batch_size=settings.places_embedding_batch_size,
            device=settings.places_embedding_device,
            text_prefix=prefix,
            normalize_embeddings=True,
        )
    raise ValueError(
        "Unsupported PLACES_EMBEDDING_PROVIDER: "
        f"{settings.places_embedding_provider}"
    )
