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
    if provider in {"sentence_transformer", "sentence-transformer", "sbert", "e5"}:
        return SentenceTransformerEmbeddingProvider(
            model_id=settings.embedding_model,
            expected_dimension=settings.embedding_dimension,
            revision=settings.sentence_transformer_revision,
            model_path=settings.sentence_transformer_model_path,
            cache_dir=settings.sentence_transformer_cache_dir,
            auto_download=settings.sentence_transformer_auto_download,
            query_prefix=settings.sentence_transformer_query_prefix,
            document_prefix=settings.sentence_transformer_document_prefix,
            batch_size=settings.sentence_transformer_batch_size,
            device=settings.sentence_transformer_device,
            max_sequence_length=settings.sentence_transformer_max_sequence_length,
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
