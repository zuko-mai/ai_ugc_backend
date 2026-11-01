"""Archive the first frame and keep each generation job replayable."""

from __future__ import annotations

import hashlib
from uuid import UUID

from app.core.logging import get_logger
from app.stages.hook_video_generator.exceptions import ComparisonNotFoundError, JobNotFoundError
from app.stages.hook_video_generator.frames import LoadedFrame
from app.stages.hook_video_generator.schemas import (
    ComparisonRecord,
    JobRecord,
    StoredGenerationRequest,
)
from app.storage.base import FileStorage

logger = get_logger(__name__)

_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class HookVideoStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def job_prefix(self, job_id: UUID) -> str:
        return f"hook_video/jobs/{job_id}"

    async def archive_frame(self, video_id: UUID, frame: LoadedFrame) -> str:
        digest = hashlib.sha256(frame.data).hexdigest()[:16]
        ext = _EXT_BY_MIME.get(frame.mime_type.lower(), ".jpg")
        key = f"hook_video/inputs/{video_id}/first_frame_{digest}{ext}"
        if await self.storage.exists(key):
            return key
        await self._save_bytes(key, frame.data, content_type=frame.mime_type)
        meta_key = f"{key}.meta.json"
        if not await self.storage.exists(meta_key):
            meta = {
                "role": "first_frame",
                "source_url": frame.source_url,
                "width": frame.width,
                "height": frame.height,
                "mime_type": frame.mime_type,
                "sha256_16": digest,
            }
            await self._save_bytes(
                meta_key,
                _dumps(meta),
                content_type="application/json",
            )
        logger.info("Archived hook video first frame video_id=%s key=%s", video_id, key)
        return key

    async def save_request(self, job_id: UUID, request: StoredGenerationRequest) -> None:
        key = f"{self.job_prefix(job_id)}/request.json"
        await self._save_bytes(
            key,
            request.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_request(self, job_id: UUID) -> StoredGenerationRequest:
        key = f"{self.job_prefix(job_id)}/request.json"
        text = await self._read_text(key, job_id=job_id)
        return StoredGenerationRequest.model_validate_json(text)

    async def save_job(self, record: JobRecord) -> None:
        key = f"{self.job_prefix(record.job_id)}/status.json"
        await self._save_bytes(
            key,
            record.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_job(self, job_id: UUID) -> JobRecord:
        key = f"{self.job_prefix(job_id)}/status.json"
        text = await self._read_text(key, job_id=job_id)
        return JobRecord.model_validate_json(text)

    async def save_output(self, job_id: UUID, data: bytes, *, extension: str) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        key = f"{self.job_prefix(job_id)}/output{ext}"
        mime = "video/quicktime" if ext == ".mov" else "video/mp4"
        await self._save_bytes(key, data, content_type=mime)
        logger.info("Stored hook video job_id=%s key=%s bytes=%s", job_id, key, len(data))
        return key

    async def save_comparison(self, record: ComparisonRecord) -> None:
        key = f"hook_video/comparisons/{record.comparison_id}.json"
        await self._save_bytes(
            key,
            record.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_comparison(self, comparison_id: UUID) -> ComparisonRecord:
        key = f"hook_video/comparisons/{comparison_id}.json"
        if not await self.storage.exists(key):
            raise ComparisonNotFoundError(
                "Hook video comparison was not found",
                details={"comparison_id": str(comparison_id)},
            )
        path = self.storage.get_local_path(key)
        return ComparisonRecord.model_validate_json(path.read_text(encoding="utf-8"))

    async def _read_text(self, key: str, *, job_id: UUID) -> str:
        if not await self.storage.exists(key):
            raise JobNotFoundError(
                "Hook video job was not found",
                details={"job_id": str(job_id)},
            )
        path = self.storage.get_local_path(key)
        return path.read_text(encoding="utf-8")

    async def _save_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        async def _stream():
            yield data

        await self.storage.save_stream(key, _stream(), content_type=content_type)


def _dumps(payload: dict) -> bytes:
    import json

    return json.dumps(payload, indent=2).encode("utf-8")
