import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles

from app.core.exceptions import AppError, StorageError
from app.core.logging import get_logger
from app.storage.base import FileStorage

logger = get_logger(__name__)


class LocalFileStorage(FileStorage):
    """Filesystem-backed storage for local development and tests."""

    def __init__(self, root: Path, public_base_url: str) -> None:
        self.root = root.resolve()
        self.public_base_url = public_base_url.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_key(self, key: str) -> Path:
        safe_key = key.lstrip("/").replace("\\", "/")
        if ".." in Path(safe_key).parts:
            raise StorageError("Invalid storage key", details={"key": key})
        path = (self.root / safe_key).resolve()
        if not path.is_relative_to(self.root):
            raise StorageError("Invalid storage key", details={"key": key})
        return path

    async def save_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        *,
        content_type: str | None = None,
    ) -> str:
        path = self._resolve_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".partial")

        try:
            written = 0
            async with aiofiles.open(tmp_path, "wb") as out:
                async for chunk in stream:
                    if not chunk:
                        continue
                    await out.write(chunk)
                    written += len(chunk)
            tmp_path.replace(path)
            logger.info(
                "Stored object key=%s bytes=%s content_type=%s",
                key,
                written,
                content_type,
            )
            return key
        except AppError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise StorageError(
                "Failed to save stream to storage",
                details={"key": key, "error": str(exc)},
            ) from exc

    async def save_file(
        self,
        key: str,
        source_path: Path,
        *,
        content_type: str | None = None,
    ) -> str:
        path = self._resolve_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(shutil.copy2, source_path, path)
            logger.info(
                "Copied file into storage key=%s source=%s content_type=%s",
                key,
                source_path,
                content_type,
            )
            return key
        except Exception as exc:
            raise StorageError(
                "Failed to copy file into storage",
                details={"key": key, "source": str(source_path), "error": str(exc)},
            ) from exc

    async def exists(self, key: str) -> bool:
        return self._resolve_key(key).is_file()

    async def delete(self, key: str) -> None:
        path = self._resolve_key(key)
        if path.is_file():
            path.unlink()
            logger.info("Deleted storage object key=%s", key)

    def get_url(self, key: str) -> str:
        return f"{self.public_base_url}/media/{key.lstrip('/')}"

    def get_local_path(self, key: str) -> Path:
        path = self._resolve_key(key)
        if not path.is_file():
            raise StorageError("Object not found in storage", details={"key": key})
        return path
