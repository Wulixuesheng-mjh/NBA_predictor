"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment or .env file."""

    # Database
    database_url: str = "postgresql+asyncpg://nba_admin:nba_predictor_2024@localhost:5432/nba_predictor"
    database_url_sync: str = "postgresql://nba_admin:nba_predictor_2024@localhost:5432/nba_predictor"

    # NBA API
    nba_api_timeout: int = 30
    nba_api_proxy: str = ""

    # Scheduler
    scheduler_enabled: bool = True
    collection_interval_minutes: int = 360  # Kaggle: every 6h (dataset updates nightly)
    balldontlie_interval_minutes: int = 30  # Real-time: every 30min (same-day scores)
    balldontlie_api_key: str = ""  # Free key from https://app.balldontlie.io

    # Model
    model_path: str = "../ml/saved_models"
    active_model_version: str = ""

    # Security
    admin_api_key: str = ""  # API Key for admin endpoints; empty = auth disabled (dev)

    # App
    app_name: str = "NBA Predictor API"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
