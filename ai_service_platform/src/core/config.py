"""
Central configuration management for AI Service Platform.
Loads settings from environment variables and config.yaml.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    name: str = Field(default="ai_platform", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="postgres", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    
    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class JWTConfig(BaseModel):
    """JWT authentication configuration."""
    secret_key: str = Field(default="your-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration in minutes")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration in days")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=100, description="Requests per minute per tenant")
    tokens_per_minute: int = Field(default=10000, description="Tokens per minute per tenant")
    burst_limit: int = Field(default=20, description="Burst limit for rate limiting")


class AIProviderConfig(BaseModel):
    """AI Provider base configuration."""
    name: str = Field(..., description="Provider name")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider")
    enabled: bool = Field(default=True, description="Whether the provider is enabled")
    models: dict = Field(default_factory=dict, description="Available models for this provider")


class PineconeConfig(BaseModel):
    """Pinecone vector database configuration."""
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(default="ai-platform", description="Default index name")
    dimension: int = Field(default=384, description="Embedding dimension")
    metric: str = Field(default="cosine", description="Similarity metric")


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="ai-platform", description="Default collection name")


class StorageConfig(BaseModel):
    """File storage configuration."""
    type: str = Field(default="local", description="Storage type (local, s3, gcs)")
    base_path: str = Field(default="./data/uploads", description="Local storage base path")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default=None, description="S3 region")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")


class BillingConfig(BaseModel):
    """Billing configuration."""
    enabled: bool = Field(default=False, description="Whether billing is enabled")
    stripe_api_key: Optional[str] = Field(default=None, description="Stripe API key")
    default_plan: str = Field(default="free", description="Default plan for new tenants")
    plans: dict = Field(default_factory=dict, description="Pricing plans configuration")


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    log_level: str = Field(default="INFO", description="Log level")
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    enable_metrics: bool = Field(default=False, description="Enable metrics")
    service_name: str = Field(default="ai-service-platform", description="Service name for tracing")


class Settings(BaseSettings):
    """Application settings."""
    
    # Environment
    env: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Core
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    # AI Providers
    providers: dict = Field(default_factory=dict, description="AI provider configurations")
    
    # Vector Databases
    pinecone: Optional[PineconeConfig] = Field(default=None, description="Pinecone configuration")
    qdrant: Optional[QdrantConfig] = Field(default=None, description="Qdrant configuration")
    
    # Storage
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Billing
    billing: BillingConfig = Field(default_factory=BillingConfig)
    
    # Observability
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    class Config:
        env_prefix = "AI_PLATFORM_"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to get settings
settings = get_settings()
