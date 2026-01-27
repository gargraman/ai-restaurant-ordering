"""Application configuration settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "hybrid-search-v2"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "admin"
    opensearch_index: str = "catering_menus"
    opensearch_use_ssl: bool = False

    # PostgreSQL / pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "hybrid_search"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    session_ttl_seconds: int = 86400  # 24 hours

    # Search Configuration
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    max_context_items: int = 8
    max_context_tokens: int = 4000
    max_per_restaurant: int = 3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_dsn(self) -> str:
        """Async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def opensearch_url(self) -> str:
        """OpenSearch URL."""
        protocol = "https" if self.opensearch_use_ssl else "http"
        return f"{protocol}://{self.opensearch_host}:{self.opensearch_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
