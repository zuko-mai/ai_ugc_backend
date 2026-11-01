"""Archive source videos and keep each text render replayable."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.core.logging import get_logger
from app.stages.text_renderer.exceptions import RenderNotFoundError
from app.stages.text_renderer.schemas import RenderRecord, StoredRenderRequest
from app.storage.base import FileStorage

logger = get_logger(__name__)


class TextRendererStore:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage

    def render_prefix(self, render_id: UUID) -> str:
        return f"text_renderer/renders/{render_id}"

    async def archive_source(
        self,
        video_id: UUID,
        data: bytes,
        *,
        suffix: str = ".mp4",
        content_type: str = "video/mp4",
    ) -> str:
        """Content-addressed archive. Never overwrites a different payload."""
        digest = hashlib.sha256(data).hexdigest()[:16]
        ext = suffix if suffix.startswith(".") else f".{suffix}"
        key = f"text_renderer/inputs/{video_id}/source_{digest}{ext}"
        if await self.storage.exists(key):
            return key
        await self._save_bytes(key, data, content_type=content_type)
        logger.info("Archived text-renderer source video_id=%s key=%s", video_id, key)
        return key

    async def save_request(self, render_id: UUID, request: StoredRenderRequest) -> None:
        key = f"{self.render_prefix(render_id)}/request.json"
        await self._save_bytes(
            key,
            request.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_request(self, render_id: UUID) -> StoredRenderRequest:
        key = f"{self.render_prefix(render_id)}/request.json"
        text = await self._read_text(key, render_id=render_id)
        return StoredRenderRequest.model_validate_json(text)

    async def save_record(self, record: RenderRecord) -> None:
        key = f"{self.render_prefix(record.render_id)}/status.json"
        await self._save_bytes(
            key,
            record.model_dump_json(indent=2).encode("utf-8"),
            content_type="application/json",
        )

    async def get_record(self, render_id: UUID) -> RenderRecord:
        key = f"{self.render_prefix(render_id)}/status.json"
        text = await self._read_text(key, render_id=render_id)
        return RenderRecord.model_validate_json(text)

    async def save_output(
        self,
        render_id: UUID,
        data: bytes,
        *,
        extension: str,
    ) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        key = f"{self.render_prefix(render_id)}/output{ext}"
        mime = "video/quicktime" if ext == ".mov" else "video/mp4"
        await self._save_bytes(key, data, content_type=mime)
        logger.info("Stored text render render_id=%s key=%s bytes=%s", render_id, key, len(data))
        return key

    async def save_preview(self, render_id: UUID, jpeg_bytes: bytes) -> str:
        key = f"{self.render_prefix(render_id)}/preview.jpg"
        await self._save_bytes(key, jpeg_bytes, content_type="image/jpeg")
        return key

    async def save_standalone_preview(self, video_id: UUID, jpeg_bytes: bytes) -> str:
        preview_id = uuid4()
        key = f"text_renderer/previews/{video_id}/{preview_id}.jpg"
        await self._save_bytes(key, jpeg_bytes, content_type="image/jpeg")
        return key

    def new_record(
        self,
        *,
        render_id: UUID,
        video_id: UUID,
        source_video_key: str,
        output_key: str,
        preview_key: str | None,
        duration_seconds: float,
        width: int,
        height: int,
        output_format: str,
        text_segments_rendered: int,
        warnings: list[str],
        parent_render_id: UUID | None = None,
    ) -> RenderRecord:
        now = datetime.now(timezone.utc).isoformat()
        return RenderRecord(
            render_id=render_id,
            video_id=video_id,
            source_video_key=source_video_key,
            output_key=output_key,
            preview_key=preview_key,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            format="mov" if output_format == "mov" else "mp4",
            text_segments_rendered=text_segments_rendered,
            warnings=warnings,
            created_at=now,
            parent_render_id=parent_render_id,
        )

    async def _read_text(self, key: str, *, render_id: UUID) -> str:
        if not await self.storage.exists(key):
            raise RenderNotFoundError(
                "Text render was not found",
                details={"render_id": str(render_id)},
            )
        path = self.storage.get_local_path(key)
        return path.read_text(encoding="utf-8")

    async def _save_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        async def _stream():
            yield data

        await self.storage.save_stream(key, _stream(), content_type=content_type)
