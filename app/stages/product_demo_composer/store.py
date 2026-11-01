"""Archive source videos and keep each compose job replayable."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import UUID

from app.core.logging import get_logger
from app.stages.product_demo_composer.exceptions import JobNotFoundError
from app.stages.product_demo_composer.schemas import JobRecord, StoredComposeRequest
from app.storage.base import FileStorage

logger = get_logger(__name__)


class ProductDemoComposerStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def job_prefix(self, job_id: UUID) -> str:
        return f"product_demo_composer/jobs/{job_id}"

    async def archive_input(
        self,
        project_id: UUID,
        role: str,
        data: bytes,
        *,
        suffix: str = ".mp4",
        content_type: str = "video/mp4",
    ) -> str:
        """Content-addressed archive. Never overwrites a different payload."""
        digest = hashlib.sha256(data).hexdigest()[:16]
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        key = f"product_demo_composer/inputs/{project_id}/{role}_{digest}{ext}"
        if await self.storage.exists(key):
            return key
        await self._save_bytes(key, data, content_type=content_type)
        logger.info(
            "Archived product-demo-composer input project_id=%s role=%s key=%s",
            project_id,
            role,
            key,
        )
        return key

    async def save_request(self, job_id: UUID, request: StoredComposeRequest) -> None:
        key = f"{self.job_prefix(job_id)}/request.json"
        await self._save_bytes(
            key,
            request.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_request(self, job_id: UUID) -> StoredComposeRequest:
        key = f"{self.job_prefix(job_id)}/request.json"
        text = await self._read_text(key, job_id=job_id)
        return StoredComposeRequest.model_validate_json(text)

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

    async def save_output(
        self,
        job_id: UUID,
        data: bytes,
        *,
        extension: str,
    ) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        key = f"{self.job_prefix(job_id)}/output{ext}"
        mime = "video/quicktime" if ext == ".mov" else "video/mp4"
        await self._save_bytes(key, data, content_type=mime)
        logger.info(
            "Stored product-demo-composer output job_id=%s key=%s bytes=%s",
            job_id,
            key,
            len(data),
        )
        return key

    def new_job(
        self,
        *,
        job_id: UUID,
        project_id: UUID,
        hook_asset_id: UUID,
        demo_asset_id: UUID,
        hook_video_key: str,
        demo_video_key: str,
        settings,
        parent_job_id: UUID | None = None,
        warnings: list[str] | None = None,
    ) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        return JobRecord(
            job_id=job_id,
            project_id=project_id,
            status="queued",
            hook_asset_id=hook_asset_id,
            demo_asset_id=demo_asset_id,
            hook_video_key=hook_video_key,
            demo_video_key=demo_video_key,
            settings=settings,
            warnings=list(warnings or []),
            parent_job_id=parent_job_id,
            created_at=now,
            updated_at=now,
        )

    async def _read_text(self, key: str, *, job_id: UUID) -> str:
        if not await self.storage.exists(key):
            raise JobNotFoundError(
                "Product demo composer job was not found",
                details={"job_id": str(job_id)},
            )
        path = self.storage.get_local_path(key)
        return path.read_text(encoding="utf-8")

    async def _save_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        async def _stream():
            yield data

        await self.storage.save_stream(key, _stream(), content_type=content_type)
