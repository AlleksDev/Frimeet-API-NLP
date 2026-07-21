from functools import lru_cache
import math
import re

from pydantic import AliasChoices, Field, model_validator
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
        default="http://3.212.166.108",
        alias="MAIN_API_BASE_URL",
    )
    main_api_places_search_path: str = Field(
        default="/api/v1/places/search",
        alias="MAIN_API_PLACES_SEARCH_PATH",
    )
    main_api_place_categories_path: str = Field(
        default="/api/v1/places/categories",
        alias="MAIN_API_PLACE_CATEGORIES_PATH",
    )
    main_api_place_catalog_language: str = Field(
        default="es",
        min_length=2,
        max_length=8,
        alias="MAIN_API_PLACE_CATALOG_LANGUAGE",
    )
    main_api_places_nearby_path: str = Field(
        default="/api/v1/places/nearby",
        alias="MAIN_API_PLACES_NEARBY_PATH",
    )
    main_api_place_anchor_resolve_path: str = Field(
        default="/api/v1/internal/places/resolve-anchor",
        alias="MAIN_API_PLACE_ANCHOR_RESOLVE_PATH",
    )
    main_api_posts_snapshot_path: str = Field(
        default="/api/v1/internal/posts/snapshot",
        alias="MAIN_API_POSTS_SNAPSHOT_PATH",
    )
    main_api_posts_changes_path: str = Field(
        default="/api/v1/internal/posts/changes",
        alias="MAIN_API_POSTS_CHANGES_PATH",
    )
    main_api_feed_interactions_path: str = Field(
        default="/api/v1/internal/feed/interactions/changes",
        alias="MAIN_API_FEED_INTERACTIONS_PATH",
    )
    main_api_users_snapshot_path: str = Field(
        default="/api/v1/internal/search/users/snapshot",
        alias="MAIN_API_USERS_SNAPSHOT_PATH",
    )
    main_api_clubs_snapshot_path: str = Field(
        default="/api/v1/internal/search/clubs/snapshot",
        alias="MAIN_API_CLUBS_SNAPSHOT_PATH",
    )
    main_api_groups_snapshot_path: str = Field(
        default="/api/v1/internal/search/groups/snapshot",
        alias="MAIN_API_GROUPS_SNAPSHOT_PATH",
    )
    main_api_events_snapshot_path: str = Field(
        default="/api/v1/internal/search/events/snapshot",
        alias="MAIN_API_EVENTS_SNAPSHOT_PATH",
    )
    main_api_internal_token: str | None = Field(
        default=None,
        alias="MAIN_API_INTERNAL_TOKEN",
    )
    main_api_auth_token: str | None = Field(default=None, alias="MAIN_API_AUTH_TOKEN")
    search_internal_token: str | None = Field(default=None, alias="SEARCH_INTERNAL_TOKEN")
    post_feed_internal_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NLP_SERVICE_TOKEN",
            "POST_FEED_INTERNAL_TOKEN",
        ),
    )
    main_api_timeout_seconds: int = Field(default=15, alias="MAIN_API_TIMEOUT_SECONDS")
    main_api_places_page_limit: int = Field(default=100, alias="MAIN_API_PLACES_PAGE_LIMIT")
    main_api_posts_page_limit: int = Field(default=100, alias="MAIN_API_POSTS_PAGE_LIMIT")
    main_api_search_page_limit: int = Field(default=100, alias="MAIN_API_SEARCH_PAGE_LIMIT")
    main_api_places_pagination_mode: str = Field(
        default="cursor",
        alias="MAIN_API_PLACES_PAGINATION_MODE",
    )
    main_api_posts_pagination_mode: str = Field(
        default="cursor",
        alias="MAIN_API_POSTS_PAGINATION_MODE",
    )
    main_api_search_pagination_mode: str = Field(
        default="cursor", alias="MAIN_API_SEARCH_PAGINATION_MODE"
    )
    global_search_nearby_boost: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        alias="GLOBAL_SEARCH_NEARBY_BOOST",
    )
    public_global_search_enabled: bool = Field(
        default=True,
        alias="PUBLIC_GLOBAL_SEARCH_ENABLED",
    )
    global_search_min_semantic_score: float = Field(
        default=0.30,
        ge=-1.0,
        le=1.0,
        alias="GLOBAL_SEARCH_MIN_SEMANTIC_SCORE",
    )
    global_search_min_lexical_score: float = Field(
        default=0.05,
        ge=0.0,
        alias="GLOBAL_SEARCH_MIN_LEXICAL_SCORE",
    )
    global_search_resource_thresholds: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        alias="GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON",
    )
    global_search_threshold_policy_version: str = Field(
        default="global-search-relevance-v1",
        min_length=1,
        max_length=64,
        alias="GLOBAL_SEARCH_THRESHOLD_POLICY_VERSION",
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
    embedding_provider: str = Field(default="fasttext", alias="EMBEDDING_PROVIDER")
    embedding_dimension: int = Field(default=300, alias="EMBEDDING_DIMENSION")
    embedding_model: str = Field(
        default="facebook/fasttext-es-vectors",
        alias="EMBEDDING_MODEL",
    )
    embedding_version: str = Field(default="common-crawl-300-v1", alias="EMBEDDING_VERSION")
    fasttext_model_path: str = Field(
        default=".models/fasttext-es/model.bin",
        alias="FASTTEXT_MODEL_PATH",
    )
    fasttext_model_repo_id: str = Field(
        default="facebook/fasttext-es-vectors",
        alias="FASTTEXT_MODEL_REPO_ID",
    )
    fasttext_model_filename: str = Field(
        default="model.bin",
        alias="FASTTEXT_MODEL_FILENAME",
    )
    fasttext_auto_download: bool = Field(
        default=True,
        alias="FASTTEXT_AUTO_DOWNLOAD",
    )

    # Places can migrate independently from posts, global search and feed
    # embeddings.  Defaults preserve the current FastText contract; enabling a
    # Sentence-Transformer is an explicit, reversible deployment choice.
    places_embedding_provider: str = Field(
        default="fasttext",
        alias="PLACES_EMBEDDING_PROVIDER",
    )
    places_embedding_dimension: int = Field(
        default=300,
        gt=0,
        alias="PLACES_EMBEDDING_DIMENSION",
    )
    places_embedding_model: str = Field(
        default="facebook/fasttext-es-vectors",
        min_length=1,
        alias="PLACES_EMBEDDING_MODEL",
    )
    places_embedding_model_revision: str | None = Field(
        default=None,
        alias="PLACES_EMBEDDING_MODEL_REVISION",
    )
    places_embedding_fix_mistral_regex: bool = Field(
        default=True,
        alias="PLACES_EMBEDDING_FIX_MISTRAL_REGEX",
    )
    places_embedding_version: str = Field(
        default="common-crawl-300-v1",
        min_length=1,
        alias="PLACES_EMBEDDING_VERSION",
    )
    places_embedding_query_prefix: str = Field(
        default="",
        alias="PLACES_EMBEDDING_QUERY_PREFIX",
    )
    places_embedding_passage_prefix: str = Field(
        default="",
        alias="PLACES_EMBEDDING_PASSAGE_PREFIX",
    )
    places_embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        alias="PLACES_EMBEDDING_BATCH_SIZE",
    )
    places_embedding_device: str | None = Field(
        default=None,
        alias="PLACES_EMBEDDING_DEVICE",
    )
    places_category_catalog_path: str | None = Field(
        default=None,
        alias="PLACES_CATEGORY_CATALOG_PATH",
    )
    places_category_min_similarity: float = Field(
        default=0.44,
        ge=-1.0,
        le=1.0,
        alias="PLACES_CATEGORY_MIN_SIMILARITY",
    )
    places_category_min_margin: float = Field(
        default=0.04,
        ge=0.0,
        le=2.0,
        alias="PLACES_CATEGORY_MIN_MARGIN",
    )
    places_pgvector_match_function: str = Field(
        default="match_places",
        min_length=1,
        alias="PLACES_PGVECTOR_MATCH_FUNCTION",
    )
    places_pgvector_hybrid_function: str | None = Field(
        default=None,
        alias="PLACES_PGVECTOR_HYBRID_FUNCTION",
    )
    places_pgvector_upsert_function: str = Field(
        default="upsert_place_embedding",
        min_length=1,
        alias="PLACES_PGVECTOR_UPSERT_FUNCTION",
    )
    places_pgvector_hash_function: str = Field(
        default="get_place_content_hashes",
        min_length=1,
        alias="PLACES_PGVECTOR_HASH_FUNCTION",
    )

    bm25_k1: float = Field(default=1.5, gt=0, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, ge=0, le=1, alias="BM25_B")
    bm25_relevance_threshold: float = Field(
        default=3.0,
        gt=0,
        alias="BM25_RELEVANCE_THRESHOLD",
    )
    semantic_no_match_threshold: float = Field(
        default=0.30,
        ge=-1,
        le=1,
        alias="SEMANTIC_NO_MATCH_THRESHOLD",
    )
    semantic_relevance_threshold: float = Field(
        default=0.50,
        ge=-1,
        le=1,
        alias="SEMANTIC_RELEVANCE_THRESHOLD",
    )
    places_chat_v2_enabled: bool = Field(
        default=False,
        alias="PLACES_CHAT_V2_ENABLED",
    )
    places_chat_llm_enabled: bool = Field(
        default=True,
        alias="PLACES_CHAT_LLM_ENABLED",
    )
    places_chat_candidate_limit: int = Field(
        default=30,
        ge=1,
        le=40,
        alias="PLACES_CHAT_CANDIDATE_LIMIT",
    )
    places_chat_min_content_score: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_MIN_CONTENT_SCORE",
    )
    places_chat_intent_min_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_INTENT_MIN_CONFIDENCE",
    )
    places_chat_ambiguity_delta: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_AMBIGUITY_DELTA",
    )
    places_chat_hypothesis_min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_HYPOTHESIS_MIN_CONFIDENCE",
    )
    places_chat_hypothesis_max_gap: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_HYPOTHESIS_MAX_GAP",
    )
    places_chat_default_radius_meters: int = Field(
        default=5_000,
        ge=1,
        le=50_000,
        alias="PLACES_CHAT_DEFAULT_RADIUS_METERS",
    )
    places_chat_max_auto_radius_meters: int = Field(
        default=50_000,
        ge=1,
        le=50_000,
        alias="PLACES_CHAT_MAX_AUTO_RADIUS_METERS",
    )
    places_chat_ranking_version: str = Field(
        default="places-chat-v3",
        min_length=1,
        max_length=64,
        alias="PLACES_CHAT_RANKING_VERSION",
    )
    places_chat_taxonomy_version: str = Field(
        default="places-taxonomy-v2",
        min_length=1,
        max_length=64,
        alias="PLACES_CHAT_TAXONOMY_VERSION",
    )
    # Contextual intent extraction is opt-in.  ``disabled`` is accepted as an
    # operational alias for the deterministic-only path so deployments can
    # explicitly turn the optional model off without changing code.
    places_chat_intent_provider: str = Field(
        default="deterministic",
        alias="PLACES_CHAT_INTENT_PROVIDER",
    )
    places_chat_bert_model_path: str | None = Field(
        default=None,
        alias="PLACES_CHAT_BERT_MODEL_PATH",
    )
    places_chat_bert_model_version: str | None = Field(
        default=None,
        alias="PLACES_CHAT_BERT_MODEL_VERSION",
    )
    places_chat_bert_device: str | None = Field(
        default=None,
        alias="PLACES_CHAT_BERT_DEVICE",
    )
    places_chat_bert_min_token_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        alias="PLACES_CHAT_BERT_MIN_TOKEN_CONFIDENCE",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout_seconds: int = Field(
        default=10,
        gt=0,
        alias="REQUEST_TIMEOUT_SECONDS",
    )
    rate_limit_requests_per_window: int = Field(
        default=600,
        ge=1,
        alias="RATE_LIMIT_REQUESTS_PER_WINDOW",
    )
    internal_rate_limit_requests_per_window: int = Field(
        default=120,
        ge=1,
        alias="INTERNAL_RATE_LIMIT_REQUESTS_PER_WINDOW",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        alias="RATE_LIMIT_WINDOW_SECONDS",
    )
    llm_timeout_seconds: int = Field(default=12, alias="LLM_TIMEOUT_SECONDS")
    max_llm_concurrent_requests: int = Field(
        default=4,
        alias="MAX_LLM_CONCURRENT_REQUESTS",
    )
    max_request_body_bytes: int = Field(
        default=262_144, ge=131_072, alias="MAX_REQUEST_BODY_BYTES"
    )
    embedding_cache_ttl_seconds: int = Field(default=300, alias="EMBEDDING_CACHE_TTL_SECONDS")
    vector_search_cache_ttl_seconds: int = Field(
        default=120,
        alias="VECTOR_SEARCH_CACHE_TTL_SECONDS",
    )
    kmeans_random_state: int = Field(default=42, alias="KMEANS_RANDOM_STATE")
    kmeans_batch_size: int = Field(default=1024, gt=0, alias="KMEANS_BATCH_SIZE")
    kmeans_min_posts: int = Field(default=200, ge=3, alias="KMEANS_MIN_POSTS")
    kmeans_min_k: int = Field(default=8, ge=2, alias="KMEANS_MIN_K")
    kmeans_max_k: int = Field(default=50, ge=2, alias="KMEANS_MAX_K")
    kmeans_lookback_days: int = Field(default=90, ge=1, alias="KMEANS_LOOKBACK_DAYS")
    kmeans_min_cluster_size: int = Field(
        default=3, ge=1, alias="KMEANS_MIN_CLUSTER_SIZE"
    )
    kmeans_max_cluster_ratio: float = Field(
        default=0.70, gt=0, le=1, alias="KMEANS_MAX_CLUSTER_RATIO"
    )
    kmeans_auto_activate: bool = Field(default=False, alias="KMEANS_AUTO_ACTIVATE")
    user_profile_half_life_days: float = Field(
        default=30.0, gt=0, alias="USER_PROFILE_HALF_LIFE_DAYS"
    )
    feed_duplicate_similarity_threshold: float = Field(
        default=0.92, ge=-1, le=1, alias="FEED_DUPLICATE_SIMILARITY_THRESHOLD"
    )
    feed_duplicate_penalty: float = Field(
        default=0.15, ge=0, le=1, alias="FEED_DUPLICATE_PENALTY"
    )

    @model_validator(mode="after")
    def validate_post_feed_security(self) -> "Settings":
        self.vector_store_provider = self.vector_store_provider.strip().lower()
        self.places_embedding_provider = self.places_embedding_provider.strip().lower()
        if self.places_embedding_model_revision is not None:
            self.places_embedding_model_revision = (
                self.places_embedding_model_revision.strip() or None
            )
        if self.places_embedding_device is not None:
            self.places_embedding_device = (
                self.places_embedding_device.strip() or None
            )
        self.places_embedding_query_prefix = _normalize_embedding_prefix(
            self.places_embedding_query_prefix
        )
        self.places_embedding_passage_prefix = _normalize_embedding_prefix(
            self.places_embedding_passage_prefix
        )
        if self.places_category_catalog_path is not None:
            self.places_category_catalog_path = (
                self.places_category_catalog_path.strip() or None
            )
        if self.places_pgvector_hybrid_function is not None:
            self.places_pgvector_hybrid_function = (
                self.places_pgvector_hybrid_function.strip() or None
            )
        if self.places_embedding_provider not in {
            "fasttext",
            "mock",
            "sentence_transformer",
            "bert",
        }:
            raise ValueError(
                "PLACES_EMBEDDING_PROVIDER debe ser fasttext, mock, "
                "sentence_transformer o bert"
            )
        if (
            self.env.lower() != "local"
            and self.places_embedding_provider in {"sentence_transformer", "bert"}
            and (
                self.places_embedding_model_revision is None
                or re.fullmatch(
                    r"[0-9a-fA-F]{40}",
                    self.places_embedding_model_revision,
                )
                is None
            )
        ):
            raise ValueError(
                "PLACES_EMBEDDING_MODEL_REVISION debe ser el SHA completo "
                "de 40 caracteres fuera del entorno local"
            )
        self.places_chat_intent_provider = (
            self.places_chat_intent_provider.strip().lower()
        )
        if self.places_chat_intent_provider not in {
            "disabled",
            "deterministic",
            "bert",
        }:
            raise ValueError(
                "PLACES_CHAT_INTENT_PROVIDER debe ser disabled, "
                "deterministic o bert"
            )
        if (
            self.places_chat_max_auto_radius_meters
            < self.places_chat_default_radius_meters
        ):
            raise ValueError(
                "PLACES_CHAT_MAX_AUTO_RADIUS_METERS debe ser mayor o igual que "
                "PLACES_CHAT_DEFAULT_RADIUS_METERS"
            )
        if self.places_chat_bert_model_path is not None:
            self.places_chat_bert_model_path = (
                self.places_chat_bert_model_path.strip() or None
            )
        if self.places_chat_bert_model_version is not None:
            self.places_chat_bert_model_version = (
                self.places_chat_bert_model_version.strip() or None
            )
        if self.places_chat_bert_device is not None:
            self.places_chat_bert_device = (
                self.places_chat_bert_device.strip() or None
            )
        if (
            self.places_chat_intent_provider == "bert"
            and self.places_chat_bert_model_path is None
        ):
            raise ValueError(
                "PLACES_CHAT_BERT_MODEL_PATH es obligatorio cuando "
                "PLACES_CHAT_INTENT_PROVIDER=bert"
            )
        allowed_resources = {"places", "posts", "users", "clubs", "groups", "events"}
        for resource_type, thresholds in self.global_search_resource_thresholds.items():
            if resource_type not in allowed_resources:
                raise ValueError(
                    f"GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON contiene un recurso invalido: {resource_type}"
                )
            if set(thresholds) != {"semantic_min", "lexical_min"}:
                raise ValueError(
                    "cada override de GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON debe "
                    "contener semantic_min y lexical_min"
                )
            semantic_min = float(thresholds["semantic_min"])
            lexical_min = float(thresholds["lexical_min"])
            if not math.isfinite(semantic_min) or not -1.0 <= semantic_min <= 1.0:
                raise ValueError("semantic_min debe ser finito y estar entre -1 y 1")
            if not math.isfinite(lexical_min) or lexical_min < 0.0:
                raise ValueError("lexical_min debe ser finito y no negativo")
        if not math.isfinite(self.global_search_min_semantic_score):
            raise ValueError("GLOBAL_SEARCH_MIN_SEMANTIC_SCORE debe ser finito")
        if not math.isfinite(self.global_search_min_lexical_score):
            raise ValueError("GLOBAL_SEARCH_MIN_LEXICAL_SCORE debe ser finito")
        self.global_search_threshold_policy_version = (
            self.global_search_threshold_policy_version.strip()
        )
        if self.env.lower() != "local":
            missing: list[str] = []
            if not self.post_feed_internal_token:
                missing.append("NLP_SERVICE_TOKEN")
            if not self.main_api_internal_token:
                missing.append("MAIN_API_INTERNAL_TOKEN")
            if missing:
                raise ValueError(
                    f"{', '.join(missing)} es obligatorio fuera del entorno local"
                )
            if self.vector_store_provider.lower() != "aws_pgvector":
                raise ValueError(
                    "VECTOR_STORE_PROVIDER=aws_pgvector es obligatorio fuera del entorno local"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _normalize_embedding_prefix(value: str) -> str:
    """Keep retrieval prefixes usable when a deployment UI trims whitespace."""

    normalized = value.strip()
    return f"{normalized} " if normalized else ""
