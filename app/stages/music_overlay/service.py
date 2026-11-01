"""Music Overlay — async mix of a user music bed onto a finished video."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.music_overlay.exceptions import (
    InvalidSettingsError,
    MediaFetchError,
    MediaTooLargeError,
    MusicOverlayError,
    OverlayError,
    ProbeError,
    UnsupportedFormatError,
)
from app.stages.music_overlay.fetch import read_media_bytes, suffix_from_url
from app.stages.music_overlay.overlay import build_plan, overlay_music
from app.stages.music_overlay.probe import probe_bytes
from app.stages.music_overlay.schemas import (
    AudioReport,
    JobRecord,
    OverlayAccepted,
    OverlayJobView,
    OverlayRequest,
    OverlayRerunRequest,
    OverlaySettings,
    StoredOverlayRequest,
    VideoAssetRef,
)
from app.stages.music_overlay.store import MusicOverlayStore
from app.storage.base import FileStorage

logger = get_logger(__name__)

_JOB_LOCKS: dict[str, asyncio.Lock] = {}
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_TERMINAL = {"success", "failed"}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".aac",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".flac",
    ".aiff",
    ".aif",
    ".m4b",
    ".caf",
    ".wma",
}


def _job_lock(job_id: UUID) -> asyncio.Lock:
    key = str(job_id)
    lock = _JOB_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _JOB_LOCKS[key] = lock
    return lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MusicOverlayService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.store = MusicOverlayStore(storage)

    async def submit(self, request: OverlayRequest) -> OverlayAccepted:
        """Validate URLs, archive originals, and queue the mix."""
        video_bytes, _video_mime, video_key, video_suffix = await self._load_url(
            project_id=request.project_id,
            role="video",
            url=request.video.video_url,
            kind="video",
        )
        music_bytes, _music_mime, music_key, music_suffix = await self._load_url(
            project_id=request.project_id,
            role="music",
            url=request.music.audio_url,
            kind="music",
        )
        return await self._accept(
            project_id=request.project_id,
            video_asset_id=request.video.asset_id,
            music_asset_id=request.music.asset_id,
            video_url=request.video.video_url,
            audio_url=request.music.audio_url,
            video_bytes=video_bytes,
            music_bytes=music_bytes,
            video_key=video_key,
            music_key=music_key,
            video_suffix=video_suffix,
            music_suffix=music_suffix,
            settings=request.settings,
        )

    async def submit_upload(
        self,
        *,
        project_id: UUID,
        video: VideoAssetRef,
        music: UploadFile,
        music_asset_id: UUID | None,
        settings: OverlaySettings,
    ) -> OverlayAccepted:
        """Same mix as submit(), with the music file taken from a multipart upload."""
        video_bytes, _mime, video_key, video_suffix = await self._load_url(
            project_id=project_id,
            role="video",
            url=video.video_url,
            kind="video",
        )
        music_bytes, music_suffix = await self._read_music_upload(music)
        music_id = music_asset_id or uuid4()
        music_key = await self.store.archive_input(
            project_id,
            "music",
            music_bytes,
            suffix=music_suffix,
            content_type=_audio_content_type(music_suffix, music.content_type),
        )
        audio_url = self.storage.get_url(music_key)
        return await self._accept(
            project_id=project_id,
            video_asset_id=video.asset_id,
            music_asset_id=music_id,
            video_url=video.video_url,
            audio_url=audio_url,
            video_bytes=video_bytes,
            music_bytes=music_bytes,
            video_key=video_key,
            music_key=music_key,
            video_suffix=video_suffix,
            music_suffix=music_suffix,
            settings=settings,
        )

    async def rerun(
        self,
        job_id: UUID,
        body: OverlayRerunRequest | None = None,
    ) -> OverlayAccepted:
        """Mix again from archived originals. Sources are not re-downloaded."""
        stored = await self.store.get_request(job_id)
        parent = await self.store.get_job(job_id)
        settings = (body.settings if body and body.settings else None) or stored.settings

        video_path = self.storage.get_local_path(stored.video_key)
        music_path = self.storage.get_local_path(stored.music_key)
        video_bytes = video_path.read_bytes()
        music_bytes = music_path.read_bytes()
        video_suffix = video_path.suffix or ".mp4"
        music_suffix = music_path.suffix or ".mp3"
        video_probe = await probe_bytes(video_bytes, suffix=video_suffix)
        music_probe = await probe_bytes(music_bytes, suffix=music_suffix)
        self._assert_usable_video(video_probe)
        self._assert_usable_music(music_probe)
        build_plan(
            video_duration=video_probe.duration_seconds,
            music_duration=music_probe.duration_seconds,
            settings=settings,
        )

        new_job_id = uuid4()
        new_stored = stored.model_copy(update={"settings": settings})
        await self.store.save_request(new_job_id, new_stored)
        record = self.store.new_job(
            job_id=new_job_id,
            project_id=stored.project_id,
            video_asset_id=stored.video_asset_id,
            music_asset_id=stored.music_asset_id,
            video_key=stored.video_key,
            music_key=stored.music_key,
            settings=settings,
            parent_job_id=parent.job_id,
            warnings=["rerun_from_archived_inputs"],
        )
        await self.store.save_job(record)
        self._schedule(new_job_id)
        logger.info(
            "Queued music overlay rerun parent=%s job_id=%s",
            job_id,
            new_job_id,
        )
        return OverlayAccepted(
            project_id=stored.project_id,
            job_id=new_job_id,
            status="queued",
            warnings=record.warnings,
        )

    async def get_job(self, job_id: UUID) -> OverlayJobView:
        """Return current status. A still-queued job is processed inline."""
        async with _job_lock(job_id):
            record = await self.store.get_job(job_id)
            if record.status in _TERMINAL:
                return self._view(record)
            if record.status == "queued":
                record = await self._process_locked(record)
            return self._view(record)

    async def get_request(self, job_id: UUID) -> StoredOverlayRequest:
        return await self.store.get_request(job_id)

    async def _accept(
        self,
        *,
        project_id: UUID,
        video_asset_id: UUID,
        music_asset_id: UUID,
        video_url: str,
        audio_url: str,
        video_bytes: bytes,
        music_bytes: bytes,
        video_key: str,
        music_key: str,
        video_suffix: str,
        music_suffix: str,
        settings: OverlaySettings,
    ) -> OverlayAccepted:
        video_probe = await probe_bytes(video_bytes, suffix=video_suffix)
        music_probe = await probe_bytes(music_bytes, suffix=music_suffix)
        self._assert_usable_video(video_probe)
        self._assert_usable_music(music_probe)
        # Fail invalid offsets before the job is accepted.
        plan = build_plan(
            video_duration=video_probe.duration_seconds,
            music_duration=music_probe.duration_seconds,
            settings=settings,
        )

        job_id = uuid4()
        stored = StoredOverlayRequest(
            project_id=project_id,
            video_asset_id=video_asset_id,
            music_asset_id=music_asset_id,
            video_url=video_url,
            audio_url=audio_url,
            video_key=video_key,
            music_key=music_key,
            settings=settings,
        )
        await self.store.save_request(job_id, stored)
        record = self.store.new_job(
            job_id=job_id,
            project_id=project_id,
            video_asset_id=video_asset_id,
            music_asset_id=music_asset_id,
            video_key=video_key,
            music_key=music_key,
            settings=settings,
            warnings=list(plan.warnings),
        )
        await self.store.save_job(record)
        self._schedule(job_id)
        logger.info(
            "Queued music overlay project_id=%s job_id=%s video=%ss music=%ss "
            "offset=%s video_start=%s",
            project_id,
            job_id,
            round(video_probe.duration_seconds, 3),
            round(music_probe.duration_seconds, 3),
            settings.music_start_offset_seconds,
            settings.video_start_seconds,
        )
        return OverlayAccepted(
            project_id=project_id,
            job_id=job_id,
            status="queued",
            warnings=list(plan.warnings),
        )

    def _schedule(self, job_id: UUID) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._safe_process(job_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    async def _safe_process(self, job_id: UUID) -> None:
        try:
            async with _job_lock(job_id):
                record = await self.store.get_job(job_id)
                if record.status != "queued":
                    return
                await self._process_locked(record)
        except Exception:  # noqa: BLE001
            logger.exception("Background music overlay failed job_id=%s", job_id)

    async def _process_locked(self, record: JobRecord) -> JobRecord:
        record.status = "processing"
        record.updated_at = _now()
        await self.store.save_job(record)
        try:
            return await self._mix_and_finalize(record)
        except (
            MediaFetchError,
            ProbeError,
            OverlayError,
            InvalidSettingsError,
            MusicOverlayError,
        ) as exc:
            record.status = "failed"
            record.error_message = exc.message
            record.updated_at = _now()
            await self.store.save_job(record)
            logger.error(
                "Music overlay failed job_id=%s error=%s details=%s",
                record.job_id,
                exc.message,
                getattr(exc, "details", None),
            )
            return record
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error_message = f"Unexpected overlay failure: {exc}"
            record.updated_at = _now()
            await self.store.save_job(record)
            logger.exception("Unexpected music overlay failure job_id=%s", record.job_id)
            return record

    async def _mix_and_finalize(self, record: JobRecord) -> JobRecord:
        video_path = self.storage.get_local_path(record.video_key)
        music_path = self.storage.get_local_path(record.music_key)
        video_bytes = video_path.read_bytes()
        music_bytes = music_path.read_bytes()
        video_suffix = video_path.suffix or ".mp4"
        music_suffix = music_path.suffix or ".mp3"

        video_probe = await probe_bytes(video_bytes, suffix=video_suffix)
        music_probe = await probe_bytes(music_bytes, suffix=music_suffix)
        self._assert_usable_video(video_probe)
        self._assert_usable_music(music_probe)

        result = await overlay_music(
            video_bytes=video_bytes,
            music_bytes=music_bytes,
            video_probe=video_probe,
            music_probe=music_probe,
            settings=record.settings,
            video_suffix=video_suffix,
            music_suffix=music_suffix,
        )

        out_probe = await probe_bytes(result.video_bytes, suffix=result.extension)
        warnings: list[str] = []
        for item in list(record.warnings) + list(result.warnings):
            if item not in warnings:
                warnings.append(item)
        if out_probe.width and out_probe.height:
            if (out_probe.width, out_probe.height) != (video_probe.width, video_probe.height):
                warnings.append(
                    "video_dimensions_changed:"
                    f"{video_probe.width}x{video_probe.height}->"
                    f"{out_probe.width}x{out_probe.height}"
                )
        if (
            out_probe.fps
            and video_probe.fps
            and abs(out_probe.fps - video_probe.fps) > 0.5
        ):
            warnings.append(
                f"video_fps_changed:{video_probe.fps:.3f}->{out_probe.fps:.3f}"
            )
        if out_probe.video_codec and video_probe.video_codec:
            if out_probe.video_codec != video_probe.video_codec:
                warnings.append(
                    "video_codec_changed:"
                    f"{video_probe.video_codec}->{out_probe.video_codec}"
                )
        duration = out_probe.duration_seconds or result.duration_seconds

        asset_id = uuid4()
        output_key = await self.store.save_output(
            record.job_id,
            result.video_bytes,
            extension=result.extension,
        )
        record.status = "success"
        record.asset_id = asset_id
        record.output_key = output_key
        record.duration_seconds = round(duration, 3)
        record.warnings = warnings
        record.error_message = None
        record.updated_at = _now()
        await self.store.save_job(record)
        logger.info(
            "Music overlay done project_id=%s job_id=%s asset_id=%s duration=%ss",
            record.project_id,
            record.job_id,
            asset_id,
            record.duration_seconds,
        )
        return record

    async def _load_url(
        self,
        *,
        project_id: UUID,
        role: str,
        url: str,
        kind: str,
    ) -> tuple[bytes, str | None, str, str]:
        max_bytes = self._max_bytes(kind)
        data, mime = await read_media_bytes(
            url,
            settings=self.settings,
            storage=self.storage,
            max_bytes=max_bytes,
            role=role,
        )
        suffix = suffix_from_url(url, mime, kind=kind)
        content_type = mime or _default_content_type(suffix, kind)
        key = await self.store.archive_input(
            project_id,
            role,
            data,
            suffix=suffix,
            content_type=content_type,
        )
        return data, mime, key, suffix

    async def _read_music_upload(self, upload: UploadFile) -> tuple[bytes, str]:
        suffix = _normalize_upload_suffix(upload.filename)
        if suffix not in AUDIO_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported music format '{suffix or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(AUDIO_EXTENSIONS))}",
                details={"extension": suffix, "allowed": sorted(AUDIO_EXTENSIONS)},
            )
        _validate_upload_mime(upload.content_type)
        data = await upload.read()
        if not data:
            raise MediaFetchError(
                "Uploaded music file is empty",
                details={"filename": upload.filename},
            )
        max_bytes = self._max_bytes("music")
        if len(data) > max_bytes:
            raise MediaTooLargeError(
                "Uploaded music file exceeds the size limit",
                details={
                    "filename": upload.filename,
                    "bytes": len(data),
                    "max_bytes": max_bytes,
                },
            )
        return data, suffix

    def _max_bytes(self, kind: str) -> int:
        if kind == "music":
            return self.settings.max_music_overlay_audio_size_mb * 1024 * 1024
        return self.settings.max_music_overlay_video_size_mb * 1024 * 1024

    def _assert_usable_video(self, probed) -> None:
        if not probed.has_video or probed.width < 2 or probed.height < 2:
            raise ProbeError(
                "Video file has no usable video stream",
                details={
                    "width": probed.width,
                    "height": probed.height,
                    "has_video": probed.has_video,
                },
            )
        if probed.duration_seconds <= 0:
            raise ProbeError(
                "Video duration is zero or unknown",
                details={"duration_seconds": probed.duration_seconds},
            )

    def _assert_usable_music(self, probed) -> None:
        if not probed.has_audio:
            raise ProbeError(
                "Music file has no audio stream",
                details={"has_audio": probed.has_audio},
            )
        if probed.duration_seconds <= 0:
            raise ProbeError(
                "Music duration is zero or unknown",
                details={"duration_seconds": probed.duration_seconds},
            )

    def _view(self, record: JobRecord) -> OverlayJobView:
        video_url = None
        asset_id = None
        duration = None
        audio = None
        if record.status == "success" and record.output_key and record.asset_id:
            video_url = self.storage.get_url(record.output_key)
            asset_id = record.asset_id
            duration = record.duration_seconds
            audio = AudioReport(
                music_start_offset_seconds=record.settings.music_start_offset_seconds,
                video_start_seconds=record.settings.video_start_seconds,
                music_volume=record.settings.music_volume,
                original_audio_enabled=record.settings.original_audio_enabled,
                fade_in_seconds=record.settings.fade_in_seconds,
                fade_out_seconds=record.settings.fade_out_seconds,
            )
        return OverlayJobView(
            project_id=record.project_id,
            job_id=record.job_id,
            asset_id=asset_id,
            video_url=video_url,
            duration_seconds=duration,
            audio=audio,
            status=record.status,
            error_message=record.error_message,
            warnings=record.warnings,
        )


def _normalize_upload_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _validate_upload_mime(content_type: str | None) -> None:
    if not content_type:
        return
    mime = content_type.split(";")[0].strip().lower()
    if mime in {"application/octet-stream", "binary/octet-stream"}:
        return
    if mime.startswith("audio/"):
        return
    if mime in {"video/mp4", "video/quicktime"}:
        return
    raise UnsupportedFormatError(
        f"Unsupported music content type '{mime}'",
        details={"content_type": mime},
    )


def _audio_content_type(suffix: str, header: str | None) -> str:
    if header:
        mime = header.split(";")[0].strip().lower()
        if mime and mime not in {"application/octet-stream", "binary/octet-stream"}:
            return mime
    return _default_content_type(suffix, "music")


def _default_content_type(suffix: str, kind: str) -> str:
    if kind == "music":
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".aac": "audio/aac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".opus": "audio/opus",
            ".flac": "audio/flac",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
        }.get(suffix, "application/octet-stream")
    if suffix == ".mov":
        return "video/quicktime"
    return "video/mp4"
