"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MedVision AI"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = Field(min_length=32)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    database_url: str = "postgresql+asyncpg://medvision:medvision@localhost:5432/medvision"
    database_url_sync: str = "postgresql://medvision:medvision@localhost:5432/medvision"

    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    storage_backend: Literal["local", "s3"] = "local"
    storage_local_path: str = "./data/storage"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "medvision-ai-data"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "medvision-brain-tumor"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"

    openai_api_key: str = ""
    google_api_key: str = ""
    llm_provider: Literal["openai", "google"] = "openai"
    llm_model: str = "gpt-4.1"
    allow_phi_to_llm: bool = False

    chroma_persist_dir: str = "./data/chroma"
    rag_embedding_model: str = "text-embedding-3-small"

    model_registry_path: str = "./data/models"
    default_model: str = "unet"
    inference_device: str = "cuda"
    use_mixed_precision: bool = True
    use_torch_compile: bool = False

    brats_data_path: str = "./datasets/brats"
    preprocessed_cache_path: str = "./data/cache"
    train_val_split: float = 0.8

    prometheus_port: int = 9090
    log_level: str = "INFO"
    log_file: str = "./logs/medvision.log"

    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        secret_key="dev-secret-key-change-in-production-min-32-chars",
        jwt_secret_key="dev-jwt-secret-key-change-in-production-min-32-chars",
    )
