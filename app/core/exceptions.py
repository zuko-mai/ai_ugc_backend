from typing import Any


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StorageError(AppError):
    """Raised on storage read/write failures."""
