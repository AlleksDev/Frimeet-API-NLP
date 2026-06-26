from functools import lru_cache

from app.shared.cache.memory import SimpleTTLCache
from app.shared.config.settings import get_settings
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.cached import CachedEmbeddingProvider
from app.shared.nlp.embeddings.mock import MockEmbeddingProvider
from app.shared.nlp.llm.base import LLMProvider
from app.shared.nlp.llm.groq_llama import GroqLlamaProvider
from app.shared.nlp.llm.mock import MockLLMProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    return CachedEmbeddingProvider(
        provider=MockEmbeddingProvider(dimension=settings.embedding_dimension),
        cache=SimpleTTLCache(default_ttl_seconds=settings.embedding_cache_ttl_seconds),
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.groq_api_key:
        return GroqLlamaProvider(settings=settings)
    return MockLLMProvider(model_name=settings.groq_model)
