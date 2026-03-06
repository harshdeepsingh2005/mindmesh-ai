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
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mindmesh:mindmesh@localhost:5432/mindmesh",
    )
    DATABASE_URL_SYNC: str = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://mindmesh:mindmesh@localhost:5432/mindmesh",
    )

    # Fallback to SQLite for local dev
    USE_SQLITE: bool = os.getenv("USE_SQLITE", "true").lower() == "true"
    SQLITE_URL: str = "sqlite+aiosqlite:///./mindmesh.db"
    SQLITE_URL_SYNC: str = "sqlite:///./mindmesh.db"

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
