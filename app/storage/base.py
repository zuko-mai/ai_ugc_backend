from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path


class FileStorage(ABC):
    """Abstract file storage shared by pipeline stages."""

    @abstractmethod
    async def save_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        *,
        content_type: str | None = None,
    ) -> str:
        """Persist a streamed object and return a storage key."""

    @abstractmethod
    async def save_file(
        self,
        key: str,
        source_path: Path,
        *,
        content_type: str | None = None,
    ) -> str:
        """Copy an existing file into storage and return a storage key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the object exists."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete the object if it exists."""

    @abstractmethod
    def get_url(self, key: str) -> str:
        """Return a URL for downstream stages."""

    @abstractmethod
    def get_local_path(self, key: str) -> Path:
        """Return a local filesystem path (materialize if remote)."""
