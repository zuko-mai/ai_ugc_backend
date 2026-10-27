from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ai-ugc-creator"
    app_env: str = "development"
    log_level: str = "INFO"

    storage_root: Path = Field(default=Path("./data/storage"))
    public_base_url: str = "http://localhost:8000"

    max_upload_size_mb: int = 500
    download_timeout_seconds: int = 120

    allowed_extensions: str = ".mp4,.mov"
    allowed_mime_types: str = "video/mp4,video/quicktime"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def extension_set(self) -> set[str]:
        return {
            ext.strip().lower()
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def mime_type_set(self) -> set[str]:
        return {
            mime.strip().lower()
            for mime in self.allowed_mime_types.split(",")
            if mime.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
