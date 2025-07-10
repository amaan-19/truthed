"""
Configuration management for the Truthed system.
Supports environment-specific settings and secret management.
"""
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator
import os


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "truthed_db"
    user: str = "truthed_user"
    password: str = "truthed_pass"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20

    @property
    def url(self) -> str:
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"

    class Config:
        env_prefix = "REDIS_"


class APISettings(BaseSettings):
    """External API configurations"""
    google_fact_check_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    news_api_key: Optional[str] = None

    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000

    class Config:
        env_prefix = "API_"


class ModelSettings(BaseSettings):
    """Model and ML configuration"""
    claim_extraction_model: str = "bert-base-uncased"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_lg"

    # Model paths
    models_dir: str = "data/models"
    checkpoints_dir: str = "data/models/checkpoints"

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_epochs: int = 10

    class Config:
        env_prefix = "MODEL_"


class Settings(BaseSettings):
    """Main application settings"""

    # Application
    app_name: str = "Truthed Professional"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # Security
    secret_key: str = "your-secret-key-change-this"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    apis: APISettings = APISettings()
    models: ModelSettings = ModelSettings()

    # Data collection
    max_articles_per_domain_per_day: int = 100
    data_retention_days: int = 365

    # Analysis thresholds
    min_claim_confidence: float = 0.7
    min_source_credibility: float = 0.3
    ensemble_weights: dict = {
        "source_credibility": 0.35,
        "claim_verification": 0.40,
        "content_quality": 0.15,
        "metadata_quality": 0.10
    }

    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()