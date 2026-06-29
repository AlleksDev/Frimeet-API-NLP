from app.shared.config.settings import Settings
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.fasttext import FastTextEmbeddingProvider
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider


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
