"""MindMesh AI — Application Configuration."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "MindMesh AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Database
    # Vercel automatically injects POSTGRES_URL when a database is linked.
    _raw_postgres_url = os.getenv("POSTGRES_URL")
    
    # Auto-convert Vercel URL to asyncpg driver if present
    if _raw_postgres_url and _raw_postgres_url.startswith("postgres://"):
        _async_url = _raw_postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
        _sync_url = _raw_postgres_url.replace("postgres://", "postgresql://", 1)
    elif _raw_postgres_url and _raw_postgres_url.startswith("postgresql://"):
        _async_url = _raw_postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        _sync_url = _raw_postgres_url
    else:
        _async_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://mindmesh:mindmesh@localhost:5432/mindmesh")
        _sync_url = os.getenv("DATABASE_URL_SYNC", "postgresql://mindmesh:mindmesh@localhost:5432/mindmesh")

    DATABASE_URL: str = _async_url
    DATABASE_URL_SYNC: str = _sync_url

    # Fallback to SQLite for local dev

    USE_SQLITE: bool = os.getenv("USE_SQLITE", "true").lower() == "true"
    SQLITE_URL: str = "sqlite+aiosqlite:////tmp/mindmesh.db"
    SQLITE_URL_SYNC: str = "sqlite:////tmp/mindmesh.db"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change_this_secret_in_production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    ALGORITHM: str = "HS256"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    @property
    def effective_database_url(self) -> str:
        """Return SQLite URL if USE_SQLITE is True, else PostgreSQL."""
        if self.USE_SQLITE:
            return self.SQLITE_URL
        return self.DATABASE_URL

    @property
    def effective_database_url_sync(self) -> str:
        """Return sync database URL for migrations."""
        if self.USE_SQLITE:
            return self.SQLITE_URL_SYNC
        return self.DATABASE_URL_SYNC

    # Models
    MODEL_SAVE_DIR: str = os.getenv("MODEL_SAVE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
