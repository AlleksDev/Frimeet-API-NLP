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

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")

    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_places_collection: str = Field(
        default="places_collection",
        alias="CHROMA_PLACES_COLLECTION",
    )
    chroma_posts_collection: str = Field(
        default="posts_collection",
        alias="CHROMA_POSTS_COLLECTION",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout_seconds: int = Field(default=10, alias="REQUEST_TIMEOUT_SECONDS")
    llm_timeout_seconds: int = Field(default=12, alias="LLM_TIMEOUT_SECONDS")
    max_llm_concurrent_requests: int = Field(
        default=4,
        alias="MAX_LLM_CONCURRENT_REQUESTS",
    )
    max_request_body_bytes: int = Field(default=65_536, alias="MAX_REQUEST_BODY_BYTES")


@lru_cache
def get_settings() -> Settings:
    return Settings()
