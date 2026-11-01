from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.pipeline.exceptions import (
    AssetNotFoundError,
    AssetUploadError,
    PipelineNotFoundError,
)
from app.pipeline.schemas import Asset, AssetKind, PipelineSession
from app.storage.base import FileStorage


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _bytes_stream(data: bytes) -> AsyncIterator[bytes]:
    yield data


class SessionStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    async def create(self) -> PipelineSession:
        now = utcnow()
        session = PipelineSession(
            session_id=uuid4(),
            created_at=now,
            updated_at=now,
        )
        await self.save(session)
        return session

    async def get(self, session_id: UUID) -> PipelineSession:
        key = self._key(session_id)
        if not await self.storage.exists(key):
            raise PipelineNotFoundError(
                "Pipeline session was not found",
                details={"session_id": str(session_id)},
            )
        return PipelineSession.model_validate_json(
            self.storage.get_local_path(key).read_text(encoding="utf-8")
        )

    async def save(self, session: PipelineSession) -> PipelineSession:
        session.updated_at = utcnow()
        data = json.dumps(session.model_dump(mode="json"), indent=2).encode("utf-8")
        await self.storage.save_stream(
            self._key(session.session_id),
            _bytes_stream(data),
            content_type="application/json",
        )
        return session

    @staticmethod
    def _key(session_id: UUID) -> str:
        return f"sessions/{session_id}/session.json"


class AssetStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    async def upload(self, upload: UploadFile, kind: AssetKind = "other") -> Asset:
        filename = Path(upload.filename or "file").name
        extension = Path(filename).suffix.lower()
        asset_id = uuid4()
        file_key = f"uploads/{asset_id}/file{extension}"
        size = 0

        async def stream() -> AsyncIterator[bytes]:
            nonlocal size
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                yield chunk

        await self.storage.save_stream(
            file_key,
            stream(),
            content_type=upload.content_type,
        )
        if size == 0:
            await self.storage.delete(file_key)
            raise AssetUploadError("Uploaded asset is empty")

        asset = Asset(
            asset_id=asset_id,
            url=self.storage.get_url(file_key),
            kind=kind,
            content_type=upload.content_type,
            filename=filename,
            size_bytes=size,
            storage_key=file_key,
            created_at=utcnow(),
        )
        await self.storage.save_stream(
            self._meta_key(asset_id),
            _bytes_stream(
                json.dumps(asset.model_dump(mode="json"), indent=2).encode("utf-8")
            ),
            content_type="application/json",
        )
        return asset

    async def get(self, asset_id: UUID) -> Asset:
        key = self._meta_key(asset_id)
        if not await self.storage.exists(key):
            raise AssetNotFoundError(
                "Asset was not found",
                details={"asset_id": str(asset_id)},
            )
        return Asset.model_validate_json(
            self.storage.get_local_path(key).read_text(encoding="utf-8")
        )

    async def as_upload(self, asset_id: UUID) -> tuple[Asset, UploadFile]:
        asset = await self.get(asset_id)
        path = self.storage.get_local_path(asset.storage_key)
        return asset, UploadFile(
            file=path.open("rb"),
            filename=asset.filename,
            headers={"content-type": asset.content_type or "application/octet-stream"},
        )

    @staticmethod
    def _meta_key(asset_id: UUID) -> str:
        return f"uploads/{asset_id}/meta.json"
