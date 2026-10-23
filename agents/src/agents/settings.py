"""Application settings via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    app_name: str = Field(default="Daily Agent Hub", validation_alias="APP_NAME")
    api_prefix: str = Field(default="/api", validation_alias="API_PREFIX")

    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_reload: bool = Field(default=False, validation_alias="API_RELOAD")

    cors_origins_raw: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173",
        validation_alias="CORS_ORIGINS",
    )

    rate_limit_default: str = Field(default="60/minute", validation_alias="RATE_LIMIT_DEFAULT")
    rate_limit_chat: str = Field(default="20/minute", validation_alias="RATE_LIMIT_CHAT")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    news_api_key: str | None = Field(default=None, validation_alias="NEWS_API_KEY")

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/agents",
        validation_alias="DATABASE_URL",
    )
    database_schema: str = Field(
        default="",
        validation_alias="DATABASE_SCHEMA",
    )
    sqlite_path: Path = Field(
        default=PROJECT_ROOT / "data" / "checkpoints.db",
        validation_alias="SQLITE_PATH",
    )
    checkpointer: Literal["memory", "sqlite", "postgres"] = Field(
        default="postgres",
        validation_alias="CHECKPOINTER",
    )

    redis_url: str | None = Field(
        default=None,
        validation_alias="REDIS_URL",
    )

    @field_validator("database_schema", mode="before")
    @classmethod
    def normalize_database_schema(cls, value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def empty_openai_base_url_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip()

    @field_validator("redis_url", mode="before")
    @classmethod
    def empty_redis_url_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip()

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: Literal["text", "json"] = Field(default="text", validation_alias="LOG_FORMAT")
    log_payloads: bool = Field(default=True, validation_alias="LOG_PAYLOADS")
    log_http_payloads: bool = Field(default=True, validation_alias="LOG_HTTP_PAYLOADS")

    # Uvicorn WebSocket keepalive (enabled by default outside development)
    ws_ping_interval: float | None = Field(
        default=None,
        validation_alias="WS_PING_INTERVAL",
    )
    ws_ping_timeout: float | None = Field(
        default=None,
        validation_alias="WS_PING_TIMEOUT",
    )
    timeout_keep_alive: int | None = Field(
        default=None,
        validation_alias="TIMEOUT_KEEP_ALIVE",
    )

    static_dir: Path = Field(
        default=PACKAGE_ROOT / "static",
        validation_alias="STATIC_DIR",
    )

    @field_validator("static_dir", mode="before")
    @classmethod
    def default_static_dir(cls, value: Path | str | None) -> Path | str:
        if value is None:
            return PACKAGE_ROOT / "static"
        if isinstance(value, str) and not value.strip():
            return PACKAGE_ROOT / "static"
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def expose_docs(self) -> bool:
        return self.app_env != "production"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def effective_database_schema(self) -> str | None:
        """Postgres schema for LangGraph tables; None uses default public."""
        name = self.database_schema.strip()
        if not name or name == "public":
            return None
        if self.database_url.lower().startswith("sqlite"):
            return None
        return name

    @property
    def redis_enabled(self) -> bool:
        return self.redis_url is not None

    @property
    def ws_keepalive_enabled(self) -> bool:
        return self.app_env != "development"

    @property
    def effective_ws_ping_interval(self) -> float | None:
        if self.ws_ping_interval is not None:
            return self.ws_ping_interval or None
        return 20.0 if self.ws_keepalive_enabled else None

    @property
    def effective_ws_ping_timeout(self) -> float | None:
        if self.ws_ping_timeout is not None:
            return self.ws_ping_timeout or None
        return 20.0 if self.ws_keepalive_enabled else None

    @property
    def effective_timeout_keep_alive(self) -> int | None:
        if self.timeout_keep_alive is not None:
            return self.timeout_keep_alive or None
        return 75 if self.ws_keepalive_enabled else None

    def uvicorn_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "factory": True,
            "host": self.api_host,
            "port": self.api_port,
            "reload": self.api_reload,
        }
        if self.effective_ws_ping_interval is not None:
            kwargs["ws_ping_interval"] = self.effective_ws_ping_interval
            kwargs["ws_ping_timeout"] = self.effective_ws_ping_timeout
        if self.effective_timeout_keep_alive is not None:
            kwargs["timeout_keep_alive"] = self.effective_timeout_keep_alive
        return kwargs


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    apply_settings_to_env(settings)
    return settings


def apply_settings_to_env(settings: Settings) -> None:
    """Expose .env values to libraries that read os.environ directly."""
    import os

    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
