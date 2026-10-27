from functools import lru_cache

from app.core.config import Settings, get_settings
from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage


@lru_cache
def get_storage() -> FileStorage:
    settings = get_settings()
    return LocalFileStorage(
        root=settings.storage_root,
        public_base_url=settings.public_base_url,
    )


def get_app_settings() -> Settings:
    return get_settings()
