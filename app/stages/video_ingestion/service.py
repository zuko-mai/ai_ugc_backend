from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.stages.video_ingestion.download import stream_from_url
from app.stages.video_ingestion.exceptions import (
    UnsupportedFormatError,
    VideoTooLargeError,
    VideoUnavailableError,
)
from app.stages.video_ingestion.social import (
    download_social_video,
    is_social_video_url,
    social_platform,
)
from app.stages.video_ingestion.metadata import extract_metadata
from app.stages.video_ingestion.schemas import VideoIngestionResponse
from app.stages.video_ingestion.validation import (
    normalize_extension,
    validate_container_magic,
    validate_extension,
    validate_mime_type,
    validate_size,
)
from app.storage.base import FileStorage

logger = get_logger(__name__)


class VideoIngestionService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    async def ingest_upload(self, upload: UploadFile) -> VideoIngestionResponse:
        extension = normalize_extension(upload.filename)
        validate_extension(extension, self.settings)
        validate_mime_type(upload.content_type, self.settings)

        video_id = uuid4()
        key = self._storage_key(video_id, extension)
        logger.info(
            "Ingesting upload video_id=%s filename=%s content_type=%s",
            video_id,
            upload.filename,
            upload.content_type,
        )

        try:
            await self.storage.save_stream(
                key,
                self._upload_stream(upload),
                content_type=upload.content_type,
            )
            return await self._finalize(video_id, key, extension)
        except (
            UnsupportedFormatError,
            VideoTooLargeError,
            VideoUnavailableError,
            StorageError,
        ):
            await self._cleanup(key)
            raise
        except Exception:
            await self._cleanup(key)
            raise

    async def ingest_url(self, url: str) -> VideoIngestionResponse:
        logger.info("Ingesting remote video url=%s", url)
        if is_social_video_url(url):
            return await self._ingest_social(url)

        extension, content_type, stream = await stream_from_url(url, self.settings)
        video_id = uuid4()
        key = self._storage_key(video_id, extension)

        try:
            await self.storage.save_stream(key, stream, content_type=content_type)
            return await self._finalize(video_id, key, extension)
        except (
            UnsupportedFormatError,
            VideoTooLargeError,
            VideoUnavailableError,
            StorageError,
        ):
            await self._cleanup(key)
            raise
        except Exception:
            await self._cleanup(key)
            raise

    async def _ingest_social(self, url: str) -> VideoIngestionResponse:
        platform = social_platform(url) or "social"
        path, extension = await download_social_video(
            url, self.settings, platform=platform
        )
        video_id = uuid4()
        key = self._storage_key(video_id, extension)
        content_type = "video/quicktime" if extension == ".mov" else "video/mp4"
        try:
            await self.storage.save_file(key, path, content_type=content_type)
            return await self._finalize(video_id, key, extension)
        except (
            UnsupportedFormatError,
            VideoTooLargeError,
            VideoUnavailableError,
            StorageError,
        ):
            await self._cleanup(key)
            raise
        except Exception:
            await self._cleanup(key)
            raise
        finally:
            try:
                path.unlink(missing_ok=True)
                path.parent.rmdir()
            except OSError:
                logger.warning("Failed to cleanup %s temp file %s", platform, path)

    async def _finalize(
        self,
        video_id: UUID,
        key: str,
        extension: str,
    ) -> VideoIngestionResponse:
        local_path = self.storage.get_local_path(key)
        await self._validate_stored_file(local_path, extension)

        metadata = await extract_metadata(local_path)
        video_url = self.storage.get_url(key)

        response = VideoIngestionResponse(
            video_id=video_id,
            video_url=video_url,
            duration_seconds=metadata.duration_seconds,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            has_audio=metadata.has_audio,
            audio_url=None,
            status="success",
            video_codec=metadata.video_codec,
            audio_codec=metadata.audio_codec,
        )
        logger.info(
            "Ingestion complete video_id=%s duration=%ss %sx%s has_audio=%s",
            video_id,
            response.duration_seconds,
            response.width,
            response.height,
            response.has_audio,
        )
        return response

    async def _validate_stored_file(self, path: Path, extension: str) -> None:
        size = path.stat().st_size
        if size == 0:
            raise VideoUnavailableError("Stored video file is empty")
        validate_size(size, self.settings)
        validate_container_magic(path.read_bytes()[:64], extension)

    async def _upload_stream(self, upload: UploadFile) -> AsyncIterator[bytes]:
        total = 0
        first_chunk: bytes | None = None
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            if first_chunk is None:
                first_chunk = chunk
                validate_container_magic(
                    chunk[:64],
                    normalize_extension(upload.filename),
                )
            total += len(chunk)
            validate_size(total, self.settings)
            yield chunk
        if total == 0:
            raise VideoUnavailableError("Uploaded video file is empty")
        logger.info(
            "Received upload stream filename=%s bytes=%s",
            upload.filename,
            total,
        )

    @staticmethod
    def _storage_key(video_id: UUID, extension: str) -> str:
        return f"originals/{video_id}/source{extension}"

    async def _cleanup(self, key: str) -> None:
        try:
            await self.storage.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cleanup storage key=%s error=%s", key, exc)
