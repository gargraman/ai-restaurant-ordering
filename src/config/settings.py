"""Application configuration settings."""

from functools import lru_cache
try:
    from typing import Literal
except ImportError:
    # For Python < 3.8, use typing_extensions
    from typing_extensions import Literal

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

    # Neo4j (Phase 2)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 30  # seconds

    # Graph Search Configuration
    graph_top_k: int = 30  # Max results from graph queries
    graph_weight: float = 1.0  # RRF weight for graph results
    graph_max_distance_km: float = 10.0  # Default radius for "nearby"
    graph_similarity_threshold: float = 0.5  # Min similarity for SIMILAR_TO

    # Feature Flags
    enable_graph_search: bool = False  # Disabled until Phase 2 implementation
    enable_3way_rrf: bool = False  # Enable 3-way RRF fusion

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Authentication / JWT
    jwt_secret_key: str = ""  # Must be set in production (min 32 chars)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_hours: int = 24
    jwt_refresh_token_expire_days: int = 7

    # Tracing (Phase 2)
    otel_tracing_enabled: bool = True
    otel_service_name: str = "hybrid-search-v2"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"

    # Stripe Payments
    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_api_version: str = "2024-12-18.acacia"

    # Payment Configuration
    platform_fee_percentage: float = 2.5  # Platform fee as percentage
    platform_fee_fixed_cents: int = 30  # Fixed fee per order in cents
    stripe_processing_fee_percentage: float = 2.9  # Stripe's fee
    stripe_processing_fee_fixed_cents: int = 30  # Stripe's fixed fee

    # Square POS Integration
    square_application_id: str = ""
    square_application_secret: str = ""
    square_environment: Literal["sandbox", "production"] = "sandbox"
    square_oauth_redirect_uri: str = ""  # e.g., https://yourdomain.com/api/pos/square/callback
    square_webhook_signature_key: str = ""

    # POS Feature Flags
    enable_pos_integration: bool = False
    enable_payments: bool = False

    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_kms_key_id: str = ""  # KMS key for encrypting POS credentials

    # SendGrid Email Notifications
    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@yourdomain.com"
    sendgrid_from_name: str = "Order Platform"
    sendgrid_from_phone: str = ""

    # SMTP Email (fallback/legacy)
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sender_email: str = ""

    # Twilio SMS Notifications
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Legacy Encryption (deprecated - use AWS KMS)
    encryption_key: str = ""

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

    def validate_required_configs(self) -> 'list[str]':
        """Validate required configuration at startup.

        Returns:
            List of missing required configuration values
        """
        errors = []

        if self.enable_payments and not self.stripe_secret_key:
            errors.append("stripe_secret_key is required when payments are enabled")

        if self.enable_payments and not self.stripe_webhook_secret:
            errors.append("stripe_webhook_secret is required when payments are enabled")

        if self.enable_pos_integration and not self.square_application_id:
            errors.append("square_application_id is required when POS integration is enabled")

        if self.enable_pos_integration and not self.square_application_secret:
            errors.append("square_application_secret is required when POS integration is enabled")

        # Check for notification configuration
        if not self.sendgrid_api_key and not self.smtp_server:
            errors.append("Either sendgrid_api_key or smtp_server must be configured for notifications")

        return errors

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

    @property
    def ENCRYPTION_KEY(self) -> str:
        """Encryption key for credential encryption."""
        return self.encryption_key


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
