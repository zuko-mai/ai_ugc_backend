"""Product Demo Composer — async [AI Hook] → [Product Demo] join service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.product_demo_composer.compose import compose_videos
from app.stages.product_demo_composer.exceptions import (
    ComposeError,
    InvalidSettingsError,
    ProbeError,
    ProductDemoComposerError,
    UnsupportedFormatError,
    VideoFetchError,
    VideoTooLargeError,
)
from app.stages.product_demo_composer.fetch import read_video_bytes, suffix_from_url
from app.stages.product_demo_composer.probe import probe_bytes
from app.stages.product_demo_composer.schemas import (
    ComposeAccepted,
    ComposeJobView,
    ComposeRerunRequest,
    ComposeSettings,
    JobRecord,
    StoredComposeRequest,
    VideoAssetRef,
)
from app.stages.product_demo_composer.store import ProductDemoComposerStore
from app.storage.base import FileStorage

logger = get_logger(__name__)

_JOB_LOCKS: dict[str, asyncio.Lock] = {}
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_TERMINAL = {"success", "failed"}


def _job_lock(job_id: UUID) -> asyncio.Lock:
    key = str(job_id)
    lock = _JOB_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _JOB_LOCKS[key] = lock
    return lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductDemoComposerService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.store = ProductDemoComposerStore(storage)

    async def submit(
        self,
        *,
        project_id: UUID,
        hook_video: VideoAssetRef,
        demo_video: UploadFile,
        demo_asset_id: UUID | None = None,
        settings: ComposeSettings | None = None,
    ) -> ComposeAccepted:
        """Validate + archive inputs, queue async composition, return job_id.

        Hook is loaded from a URL. Demo is accepted as a multipart file upload.
        """
        compose_settings = settings or ComposeSettings()
        demo_id = demo_asset_id or uuid4()

        hook_bytes, _hook_mime, hook_key, hook_suffix = await self._load_and_archive(
            project_id=project_id,
            role="hook",
            url=hook_video.video_url,
        )
        demo_bytes, demo_mime, demo_key, demo_suffix = await self._load_and_archive_upload(
            project_id=project_id,
            role="demo",
            upload=demo_video,
        )
        demo_url = self.storage.get_url(demo_key)

        # Probe early so invalid media fails before the job is accepted.
        hook_probe = await probe_bytes(hook_bytes, suffix=hook_suffix)
        demo_probe = await probe_bytes(demo_bytes, suffix=demo_suffix)
        self._assert_usable_video(hook_probe, role="hook")
        self._assert_usable_video(demo_probe, role="demo")

        warnings: list[str] = []
        if not hook_probe.has_audio:
            warnings.append("hook_missing_audio")
        if not demo_probe.has_audio:
            warnings.append("demo_missing_audio")

        job_id = uuid4()
        stored = StoredComposeRequest(
            project_id=project_id,
            hook_asset_id=hook_video.asset_id,
            demo_asset_id=demo_id,
            hook_video_url=hook_video.video_url,
            demo_video_url=demo_url,
            hook_video_key=hook_key,
            demo_video_key=demo_key,
            settings=compose_settings,
        )
        await self.store.save_request(job_id, stored)

        record = self.store.new_job(
            job_id=job_id,
            project_id=project_id,
            hook_asset_id=hook_video.asset_id,
            demo_asset_id=demo_id,
            hook_video_key=hook_key,
            demo_video_key=demo_key,
            settings=compose_settings,
            warnings=warnings,
        )
        await self.store.save_job(record)
        self._schedule(job_id)

        logger.info(
            "Queued product-demo compose project_id=%s job_id=%s hook=%ss demo=%ss "
            "demo_file=%s",
            project_id,
            job_id,
            round(hook_probe.duration_seconds, 3),
            round(demo_probe.duration_seconds, 3),
            demo_video.filename,
        )
        return ComposeAccepted(
            project_id=project_id,
            job_id=job_id,
            status="queued",
            warnings=warnings,
        )

    async def submit_refs(
        self,
        *,
        project_id: UUID,
        hook_video: VideoAssetRef,
        demo_video: VideoAssetRef,
        settings: ComposeSettings | None = None,
    ) -> ComposeAccepted:
        """Compose from already-stored hook + demo URLs (no re-upload)."""
        compose_settings = settings or ComposeSettings()

        hook_bytes, _hook_mime, hook_key, hook_suffix = await self._load_and_archive(
            project_id=project_id,
            role="hook",
            url=hook_video.video_url,
        )
        demo_bytes, _demo_mime, demo_key, demo_suffix = await self._load_and_archive(
            project_id=project_id,
            role="demo",
            url=demo_video.video_url,
        )

        hook_probe = await probe_bytes(hook_bytes, suffix=hook_suffix)
        demo_probe = await probe_bytes(demo_bytes, suffix=demo_suffix)
        self._assert_usable_video(hook_probe, role="hook")
        self._assert_usable_video(demo_probe, role="demo")

        warnings: list[str] = []
        if not hook_probe.has_audio:
            warnings.append("hook_missing_audio")
        if not demo_probe.has_audio:
            warnings.append("demo_missing_audio")

        job_id = uuid4()
        stored = StoredComposeRequest(
            project_id=project_id,
            hook_asset_id=hook_video.asset_id,
            demo_asset_id=demo_video.asset_id,
            hook_video_url=hook_video.video_url,
            demo_video_url=demo_video.video_url,
            hook_video_key=hook_key,
            demo_video_key=demo_key,
            settings=compose_settings,
        )
        await self.store.save_request(job_id, stored)

        record = self.store.new_job(
            job_id=job_id,
            project_id=project_id,
            hook_asset_id=hook_video.asset_id,
            demo_asset_id=demo_video.asset_id,
            hook_video_key=hook_key,
            demo_video_key=demo_key,
            settings=compose_settings,
            warnings=warnings,
        )
        await self.store.save_job(record)
        self._schedule(job_id)

        logger.info(
            "Queued product-demo compose from refs project_id=%s job_id=%s",
            project_id,
            job_id,
        )
        return ComposeAccepted(
            project_id=project_id,
            job_id=job_id,
            status="queued",
            warnings=warnings,
        )

    async def rerun(
        self,
        job_id: UUID,
        body: ComposeRerunRequest | None = None,
    ) -> ComposeAccepted:
        """Recompose from archived originals with optional settings override."""
        stored = await self.store.get_request(job_id)
        parent = await self.store.get_job(job_id)
        settings = (body.settings if body and body.settings else None) or stored.settings

        new_job_id = uuid4()
        new_stored = StoredComposeRequest(
            project_id=stored.project_id,
            hook_asset_id=stored.hook_asset_id,
            demo_asset_id=stored.demo_asset_id,
            hook_video_url=stored.hook_video_url,
            demo_video_url=stored.demo_video_url,
            hook_video_key=stored.hook_video_key,
            demo_video_key=stored.demo_video_key,
            settings=settings,
        )
        await self.store.save_request(new_job_id, new_stored)

        record = self.store.new_job(
            job_id=new_job_id,
            project_id=stored.project_id,
            hook_asset_id=stored.hook_asset_id,
            demo_asset_id=stored.demo_asset_id,
            hook_video_key=stored.hook_video_key,
            demo_video_key=stored.demo_video_key,
            settings=settings,
            parent_job_id=parent.job_id,
            warnings=["rerun_from_archived_inputs"],
        )
        await self.store.save_job(record)
        self._schedule(new_job_id)

        logger.info(
            "Queued product-demo recomposition parent=%s job_id=%s",
            job_id,
            new_job_id,
        )
        return ComposeAccepted(
            project_id=stored.project_id,
            job_id=new_job_id,
            status="queued",
            warnings=record.warnings,
        )

    async def get_job(self, job_id: UUID) -> ComposeJobView:
        """Return current status. If still queued, process inline (work-steal)."""
        async with _job_lock(job_id):
            record = await self.store.get_job(job_id)
            if record.status in _TERMINAL:
                return self._view(record)
            if record.status == "queued":
                record = await self._process_locked(record)
            return self._view(record)

    async def get_request(self, job_id: UUID) -> StoredComposeRequest:
        return await self.store.get_request(job_id)

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
            logger.exception("Background product-demo compose failed job_id=%s", job_id)

    async def _process_locked(self, record: JobRecord) -> JobRecord:
        record.status = "processing"
        record.updated_at = _now()
        await self.store.save_job(record)

        try:
            return await self._compose_and_finalize(record)
        except (
            VideoFetchError,
            ProbeError,
            ComposeError,
            InvalidSettingsError,
            ProductDemoComposerError,
        ) as exc:
            record.status = "failed"
            record.error_message = exc.message
            record.updated_at = _now()
            await self.store.save_job(record)
            logger.error(
                "Product-demo compose failed job_id=%s error=%s details=%s",
                record.job_id,
                exc.message,
                getattr(exc, "details", None),
            )
            return record
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error_message = f"Unexpected compose failure: {exc}"
            record.updated_at = _now()
            await self.store.save_job(record)
            logger.exception("Unexpected product-demo compose failure job_id=%s", record.job_id)
            return record

    async def _compose_and_finalize(self, record: JobRecord) -> JobRecord:
        hook_path = self.storage.get_local_path(record.hook_video_key)
        demo_path = self.storage.get_local_path(record.demo_video_key)
        hook_bytes = hook_path.read_bytes()
        demo_bytes = demo_path.read_bytes()
        hook_suffix = hook_path.suffix or ".mp4"
        demo_suffix = demo_path.suffix or ".mp4"

        hook_probe = await probe_bytes(hook_bytes, suffix=hook_suffix)
        demo_probe = await probe_bytes(demo_bytes, suffix=demo_suffix)
        self._assert_usable_video(hook_probe, role="hook")
        self._assert_usable_video(demo_probe, role="demo")

        result = await compose_videos(
            hook_bytes=hook_bytes,
            demo_bytes=demo_bytes,
            hook_probe=hook_probe,
            demo_probe=demo_probe,
            settings=record.settings,
            hook_suffix=hook_suffix,
            demo_suffix=demo_suffix,
        )

        # Confirm output geometry with ffprobe.
        out_probe = await probe_bytes(result.video_bytes, suffix=result.extension)
        warnings = list(record.warnings) + list(result.warnings)
        if out_probe.width and out_probe.height:
            if (out_probe.width, out_probe.height) != (result.width, result.height):
                warnings.append(
                    f"output_dimensions_changed:"
                    f"{result.width}x{result.height}->{out_probe.width}x{out_probe.height}"
                )
        duration = out_probe.duration_seconds or result.duration_seconds
        width = out_probe.width or result.width
        height = out_probe.height or result.height
        fps = out_probe.fps or result.fps

        # Adjust segment end to measured duration when close.
        segments = list(result.segments)
        if segments and duration > 0:
            segments[-1] = segments[-1].model_copy(update={"end_seconds": round(duration, 3)})

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
        record.width = width
        record.height = height
        record.fps = round(float(fps), 3)
        record.segments = segments
        record.warnings = warnings
        record.error_message = None
        record.updated_at = _now()
        await self.store.save_job(record)

        logger.info(
            "Product-demo compose done project_id=%s job_id=%s asset_id=%s "
            "duration=%ss %sx%s",
            record.project_id,
            record.job_id,
            asset_id,
            record.duration_seconds,
            width,
            height,
        )
        return record

    async def _load_and_archive(
        self,
        *,
        project_id: UUID,
        role: str,
        url: str,
    ) -> tuple[bytes, str | None, str, str]:
        max_bytes = self.settings.max_product_demo_video_size_mb * 1024 * 1024
        data, mime = await read_video_bytes(
            url,
            settings=self.settings,
            storage=self.storage,
            max_bytes=max_bytes,
        )
        suffix = suffix_from_url(url, mime)
        content_type = mime or ("video/quicktime" if suffix == ".mov" else "video/mp4")
        key = await self.store.archive_input(
            project_id,
            role,
            data,
            suffix=suffix,
            content_type=content_type,
        )
        return data, mime, key, suffix

    async def _load_and_archive_upload(
        self,
        *,
        project_id: UUID,
        role: str,
        upload: UploadFile,
    ) -> tuple[bytes, str | None, str, str]:
        suffix = _normalize_upload_suffix(upload.filename)
        _validate_upload_extension(suffix, self.settings)
        _validate_upload_mime(upload.content_type, self.settings)

        data = await upload.read()
        if not data:
            raise VideoFetchError(
                "Uploaded demo video is empty",
                details={"filename": upload.filename},
            )
        max_bytes = self.settings.max_product_demo_video_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise VideoTooLargeError(
                "Uploaded demo video exceeds the size limit",
                details={
                    "filename": upload.filename,
                    "bytes": len(data),
                    "max_bytes": max_bytes,
                },
            )
        _validate_container_magic(data[:64], suffix)

        mime = (upload.content_type or "").split(";")[0].strip().lower() or None
        content_type = mime or ("video/quicktime" if suffix == ".mov" else "video/mp4")
        key = await self.store.archive_input(
            project_id,
            role,
            data,
            suffix=suffix,
            content_type=content_type,
        )
        return data, mime, key, suffix

    def _assert_usable_video(self, probed, *, role: str) -> None:
        if not probed.has_video or probed.width < 2 or probed.height < 2:
            raise ProbeError(
                f"{role.capitalize()} file has no usable video stream",
                details={
                    "role": role,
                    "width": probed.width,
                    "height": probed.height,
                    "has_video": probed.has_video,
                },
            )
        if probed.duration_seconds <= 0:
            raise ProbeError(
                f"{role.capitalize()} video duration is zero or unknown",
                details={"role": role, "duration_seconds": probed.duration_seconds},
            )

    def _view(self, record: JobRecord) -> ComposeJobView:
        video_url = None
        if record.status == "success" and record.output_key:
            video_url = self.storage.get_url(record.output_key)
        return ComposeJobView(
            project_id=record.project_id,
            job_id=record.job_id,
            asset_id=record.asset_id,
            video_url=video_url,
            duration_seconds=record.duration_seconds,
            width=record.width,
            height=record.height,
            fps=record.fps,
            segments=record.segments,
            status=record.status,
            error_message=record.error_message,
            warnings=record.warnings,
        )


def _normalize_upload_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _validate_upload_extension(extension: str, settings: Settings) -> None:
    if extension not in settings.extension_set:
        raise UnsupportedFormatError(
            f"Unsupported demo video format '{extension or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(settings.extension_set))}",
            details={"extension": extension, "allowed": sorted(settings.extension_set)},
        )


def _validate_upload_mime(content_type: str | None, settings: Settings) -> None:
    if not content_type:
        return
    mime = content_type.split(";")[0].strip().lower()
    if mime in {"application/octet-stream", "binary/octet-stream"}:
        return
    if mime not in settings.mime_type_set:
        raise UnsupportedFormatError(
            f"Unsupported demo content type '{mime}'. "
            f"Allowed: {', '.join(sorted(settings.mime_type_set))}",
            details={"content_type": mime, "allowed": sorted(settings.mime_type_set)},
        )


def _validate_container_magic(header: bytes, extension: str) -> None:
    if extension not in {".mp4", ".mov", ".m4v"}:
        return
    marker = header[4:8] if len(header) >= 8 else b""
    if marker != b"ftyp" and b"ftyp" not in header[:64]:
        raise UnsupportedFormatError(
            "Uploaded file does not look like a valid MP4/MOV container",
            details={"extension": extension},
        )
