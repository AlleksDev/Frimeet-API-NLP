from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="local", alias="ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")

    main_api_base_url: str = Field(
        default="http://52.86.8.11",
        alias="MAIN_API_BASE_URL",
    )
    main_api_places_search_path: str = Field(
        default="/api/v1/places/search",
        alias="MAIN_API_PLACES_SEARCH_PATH",
    )
    main_api_places_nearby_path: str = Field(
        default="/api/v1/places/nearby",
        alias="MAIN_API_PLACES_NEARBY_PATH",
    )
    main_api_posts_search_path: str = Field(
        default="/api/v1/posts/search",
        alias="MAIN_API_POSTS_SEARCH_PATH",
    )
    main_api_internal_token: str | None = Field(
        default=None,
        alias="MAIN_API_INTERNAL_TOKEN",
    )
    main_api_auth_token: str | None = Field(default=None, alias="MAIN_API_AUTH_TOKEN")
    main_api_timeout_seconds: int = Field(default=15, alias="MAIN_API_TIMEOUT_SECONDS")
    main_api_places_page_limit: int = Field(default=100, alias="MAIN_API_PLACES_PAGE_LIMIT")
    main_api_posts_page_limit: int = Field(default=100, alias="MAIN_API_POSTS_PAGE_LIMIT")
    main_api_places_pagination_mode: str = Field(
        default="cursor",
        alias="MAIN_API_PLACES_PAGINATION_MODE",
    )
    main_api_posts_pagination_mode: str = Field(
        default="cursor",
        alias="MAIN_API_POSTS_PAGINATION_MODE",
    )

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")

    vector_store_provider: str = Field(default="mock", alias="VECTOR_STORE_PROVIDER")

    pgvector_host: str | None = Field(default=None, alias="PGVECTOR_HOST")
    pgvector_port: int = Field(default=5432, alias="PGVECTOR_PORT")
    pgvector_database: str | None = Field(default=None, alias="PGVECTOR_DATABASE")
    pgvector_user: str | None = Field(default=None, alias="PGVECTOR_USER")
    pgvector_password: str | None = Field(default=None, alias="PGVECTOR_PASSWORD")
    pgvector_reader_user: str | None = Field(default=None, alias="PGVECTOR_READER_USER")
    pgvector_reader_password: str | None = Field(
        default=None,
        alias="PGVECTOR_READER_PASSWORD",
    )
    pgvector_writer_user: str | None = Field(default=None, alias="PGVECTOR_WRITER_USER")
    pgvector_writer_password: str | None = Field(
        default=None,
        alias="PGVECTOR_WRITER_PASSWORD",
    )
    pgvector_ssl_mode: str = Field(default="require", alias="PGVECTOR_SSL_MODE")
    pgvector_places_table: str = "place_embeddings"
    pgvector_posts_table: str = "post_embeddings"
    embedding_dimension: int = Field(default=16, alias="EMBEDDING_DIMENSION")
    embedding_model: str = Field(default="mock-embedding", alias="EMBEDDING_MODEL")
    embedding_version: str = Field(default="v1", alias="EMBEDDING_VERSION")

    bm25_k1: float = Field(default=1.5, gt=0, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, ge=0, le=1, alias="BM25_B")
    bm25_relevance_threshold: float = Field(
        default=3.0,
        gt=0,
        alias="BM25_RELEVANCE_THRESHOLD",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout_seconds: int = Field(default=10, alias="REQUEST_TIMEOUT_SECONDS")
    llm_timeout_seconds: int = Field(default=12, alias="LLM_TIMEOUT_SECONDS")
    max_llm_concurrent_requests: int = Field(
        default=4,
        alias="MAX_LLM_CONCURRENT_REQUESTS",
    )
    max_request_body_bytes: int = Field(default=65_536, alias="MAX_REQUEST_BODY_BYTES")
    embedding_cache_ttl_seconds: int = Field(default=300, alias="EMBEDDING_CACHE_TTL_SECONDS")
    vector_search_cache_ttl_seconds: int = Field(
        default=120,
        alias="VECTOR_SEARCH_CACHE_TTL_SECONDS",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
