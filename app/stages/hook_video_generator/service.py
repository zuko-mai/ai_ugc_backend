"""Queue image-to-video jobs and publish a file only after it is validated."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.hook_video_generator.exceptions import (
    GenerationFailedError,
    HookVideoError,
    PayloadFetchError,
    ProbeError,
    ProviderApiError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from app.stages.hook_video_generator.fetch import read_url_bytes
from app.stages.hook_video_generator.frames import (
    LoadedFrame,
    load_archived_frame,
    load_frame,
    resolution_warnings,
)
from app.stages.hook_video_generator.motion_prompt import compile_motion_instructions
from app.stages.hook_video_generator.normalize import normalize_video_cover
from app.stages.hook_video_generator.probe import probe_bytes
from app.stages.hook_video_generator.providers.base import (
    ProviderImage,
    VideoProviderRequest,
    normalize_remote_status,
)
from app.stages.hook_video_generator.providers.registry import get_provider
from app.stages.hook_video_generator.schemas import (
    ComparisonRecord,
    HookVideoAccepted,
    HookVideoCallback,
    HookVideoCompareAccepted,
    HookVideoCompareRequest,
    HookVideoComparisonView,
    HookVideoGenerateRequest,
    HookVideoJobView,
    HookVideoRerunRequest,
    JobRecord,
    StoredGenerationRequest,
    parse_resolution,
)
from app.stages.hook_video_generator.store import HookVideoStore
from app.stages.hook_video_generator.validation import assess_output
from app.storage.base import FileStorage

logger = get_logger(__name__)

_JOB_LOCKS: dict[str, asyncio.Lock] = {}
_TERMINAL = {"success", "failed"}
_OPEN_SUBMIT = {"queued", "processing"}


def _job_lock(job_id: UUID) -> asyncio.Lock:
    key = str(job_id)
    lock = _JOB_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _JOB_LOCKS[key] = lock
    return lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HookVideoService:
    def __init__(self, storage: FileStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self.store = HookVideoStore(storage)

    async def submit(self, request: HookVideoGenerateRequest) -> HookVideoAccepted:
        frame = await load_frame(
            request.first_frame_url,
            settings=self.settings,
            storage=self.storage,
        )
        return await self._enqueue(request, frame)

    async def rerun(
        self,
        job_id: UUID,
        override: HookVideoRerunRequest | None = None,
    ) -> HookVideoAccepted:
        stored = await self.store.get_request(job_id)
        frame = await load_archived_frame(
            storage=self.storage,
            key=stored.first_frame_key,
            source_url=stored.first_frame_url,
        )
        override = override or HookVideoRerunRequest()
        motion_plan = stored.motion_plan
        if override.motion_plan is not None:
            motion_plan = override.motion_plan
        elif override.motion_prompt is not None:
            motion_plan = {**stored.motion_plan, "motionPrompt": override.motion_prompt}
        request = HookVideoGenerateRequest(
            video_id=stored.video_id,
            first_frame_url=stored.first_frame_url,
            motion_plan=motion_plan,
            duration_seconds=override.duration_seconds or stored.duration_seconds,
            aspect_ratio=stored.aspect_ratio,  # type: ignore[arg-type]
            output_resolution=stored.output_resolution,
            fps=stored.fps,
            model_provider=override.model_provider or stored.model_provider,
            model_name=override.model_name or stored.model_name,
        )
        return await self._enqueue(request, frame)

    async def compare(self, request: HookVideoCompareRequest) -> HookVideoCompareAccepted:
        for target in request.models:
            get_provider(target.model_provider, self.settings)
        frame = await load_frame(
            request.first_frame_url,
            settings=self.settings,
            storage=self.storage,
        )
        comparison_id = uuid4()
        jobs: list[HookVideoAccepted] = []
        for target in request.models:
            generate = HookVideoGenerateRequest(
                video_id=request.video_id,
                first_frame_url=request.first_frame_url,
                motion_plan=request.motion_plan,
                duration_seconds=request.duration_seconds,
                aspect_ratio=request.aspect_ratio,
                output_resolution=request.output_resolution,
                fps=request.fps,
                model_provider=target.model_provider,
                model_name=target.model_name,
            )
            jobs.append(await self._enqueue(generate, frame, comparison_id=comparison_id))
        await self.store.save_comparison(
            ComparisonRecord(
                comparison_id=comparison_id,
                video_id=request.video_id,
                job_ids=[item.job_id for item in jobs],
            )
        )
        return HookVideoCompareAccepted(
            comparison_id=comparison_id,
            video_id=request.video_id,
            jobs=jobs,
        )

    async def get_job(self, job_id: UUID) -> HookVideoJobView:
        async with _job_lock(job_id):
            record = await self.store.get_job(job_id)
            if record.status in _TERMINAL:
                return self._view(record)
            return await self._advance(record)

    async def get_comparison(self, comparison_id: UUID) -> HookVideoComparisonView:
        record = await self.store.get_comparison(comparison_id)
        jobs = [await self.get_job(job_id) for job_id in record.job_ids]
        return HookVideoComparisonView(
            comparison_id=record.comparison_id,
            video_id=record.video_id,
            jobs=jobs,
        )

    async def get_request(self, job_id: UUID) -> StoredGenerationRequest:
        return await self.store.get_request(job_id)

    async def handle_callback(
        self,
        job_id: UUID,
        callback: HookVideoCallback,
    ) -> HookVideoJobView:
        try:
            remote_status = normalize_remote_status(callback.provider_status)
        except ValueError as exc:
            raise HookVideoError(
                f"Unrecognized callback status '{callback.provider_status}'",
                details={"provider_status": callback.provider_status},
            ) from exc

        async with _job_lock(job_id):
            record = await self.store.get_job(job_id)
            if record.status in _TERMINAL:
                return self._view(record)
            if remote_status == "failed":
                record.status = "failed"
                record.error_message = callback.error or "Provider reported failure"
                return await self._persist(record)
            if remote_status == "succeeded" and callback.video_url:
                return await self._finalize(
                    record,
                    video_bytes=None,
                    video_url=callback.video_url,
                    extra_warnings=["completion_via_callback"],
                )
            return await self._advance(record)

    async def _enqueue(
        self,
        request: HookVideoGenerateRequest,
        frame: LoadedFrame,
        *,
        comparison_id: UUID | None = None,
    ) -> HookVideoAccepted:
        provider_name = (
            request.model_provider or self.settings.hook_video_provider or "kling"
        ).strip().lower()
        model_name = (
            request.model_name or self.settings.hook_video_model or "kling-3.0"
        ).strip()
        provider = get_provider(provider_name, self.settings)
        width, height = parse_resolution(request.output_resolution)
        instructions = compile_motion_instructions(
            request.motion_plan,
            duration_seconds=request.duration_seconds,
        )
        frame_key = await self.store.archive_frame(request.video_id, frame)
        warnings = list(instructions.warnings)
        warnings.extend(
            resolution_warnings(
                frame_width=frame.width,
                frame_height=frame.height,
                output_width=width,
                output_height=height,
            )
        )
        job_id = uuid4()
        created_at = _now()
        await self.store.save_request(
            job_id,
            StoredGenerationRequest(
                video_id=request.video_id,
                first_frame_url=request.first_frame_url,
                first_frame_key=frame_key,
                motion_plan=request.motion_plan,
                duration_seconds=request.duration_seconds,
                aspect_ratio=request.aspect_ratio,
                output_resolution=request.output_resolution,
                width=width,
                height=height,
                fps=request.fps,
                model_provider=provider_name,
                model_name=model_name,
                compiled_prompt=instructions.prompt,
                negative_prompt=instructions.negative_prompt,
                identity_lock=instructions.identity_lock,
                camera_lock=instructions.camera_lock,
            ),
        )
        provider_request = VideoProviderRequest(
            first_frame=ProviderImage(
                data=frame.data,
                mime_type=frame.mime_type,
                width=frame.width,
                height=frame.height,
                source_url=frame.source_url,
            ),
            instructions=instructions,
            duration_seconds=request.duration_seconds,
            aspect_ratio=request.aspect_ratio,
            width=width,
            height=height,
            fps=request.fps,
            model_name=model_name,
            timeout_seconds=float(self.settings.hook_video_timeout_seconds),
        )
        try:
            submission = await provider.submit(provider_request)
        except (ProviderApiError, ProviderTimeoutError) as exc:
            await self._persist(
                self._record(
                    job_id=job_id,
                    request=request,
                    provider_name=provider_name,
                    model_name=model_name,
                    frame_key=frame_key,
                    width=width,
                    height=height,
                    warnings=warnings,
                    comparison_id=comparison_id,
                    created_at=created_at,
                    status="failed",
                    error_message=exc.message,
                )
            )
            details = dict(exc.details)
            details["job_id"] = str(job_id)
            raise type(exc)(exc.message, details=details) from exc

        if submission.status not in _OPEN_SUBMIT or not submission.provider_job_id:
            message = (
                "Provider submit returned a terminal status. "
                "The task stays incomplete until poll or callback provides a video file "
                "that passes validation."
            )
            await self._persist(
                self._record(
                    job_id=job_id,
                    request=request,
                    provider_name=provider_name,
                    model_name=model_name,
                    frame_key=frame_key,
                    width=width,
                    height=height,
                    warnings=warnings,
                    comparison_id=comparison_id,
                    created_at=created_at,
                    status="failed",
                    provider_job_id=submission.provider_job_id or None,
                    error_message=message,
                )
            )
            raise GenerationFailedError(
                message,
                details={"job_id": str(job_id), "status": submission.status},
            )

        _extend(warnings, submission.provider_warnings)
        record = self._record(
            job_id=job_id,
            request=request,
            provider_name=provider_name,
            model_name=model_name,
            frame_key=frame_key,
            width=width,
            height=height,
            warnings=warnings,
            comparison_id=comparison_id,
            created_at=created_at,
            status=submission.status,
            provider_job_id=submission.provider_job_id,
        )
        await self._persist(record)
        logger.info(
            "Hook video queued job_id=%s video_id=%s provider=%s model=%s",
            job_id,
            request.video_id,
            provider_name,
            model_name,
        )
        return HookVideoAccepted(
            job_id=job_id,
            video_id=request.video_id,
            status=submission.status,  # type: ignore[arg-type]
            model_provider=provider_name,
            model_name=model_name,
            warnings=list(record.warnings),
            comparison_id=comparison_id,
        )

    async def _advance(self, record: JobRecord) -> HookVideoJobView:
        if not record.provider_job_id:
            record.status = "failed"
            record.error_message = "Job has no provider task id"
            return await self._persist(record)
        provider = get_provider(record.model_provider, self.settings)
        try:
            poll = await provider.poll(record.provider_job_id)
        except ProviderTimeoutError as exc:
            _extend(record.warnings, [f"provider_timeout: {exc.message}"])
            return await self._persist(record)
        except (ProviderApiError, ProviderNotConfiguredError) as exc:
            record.status = "failed"
            record.error_message = exc.message
            return await self._persist(record)

        if poll.status in _OPEN_SUBMIT:
            record.status = poll.status
            _extend(record.warnings, poll.provider_warnings)
            return await self._persist(record)
        if poll.status == "failed":
            record.status = "failed"
            record.error_message = poll.error_message or "Provider failed"
            _extend(record.warnings, poll.provider_warnings)
            return await self._persist(record)
        if poll.status == "succeeded":
            return await self._finalize(
                record,
                video_bytes=poll.video_bytes,
                video_url=poll.video_url,
                extra_warnings=list(poll.provider_warnings),
            )
        record.status = "failed"
        record.error_message = f"Unexpected provider status '{poll.status}'"
        return await self._persist(record)

    async def _finalize(
        self,
        record: JobRecord,
        *,
        video_bytes: bytes | None,
        video_url: str | None,
        extra_warnings: list[str],
    ) -> HookVideoJobView:
        _extend(record.warnings, extra_warnings)
        if video_bytes is None:
            if not video_url:
                record.status = "failed"
                record.error_message = "Provider completed without a video payload"
                return await self._persist(record)
            try:
                video_bytes, _mime = await read_url_bytes(
                    video_url,
                    settings=self.settings,
                    storage=self.storage,
                    max_bytes=self.settings.max_upload_size_bytes,
                )
            except PayloadFetchError as exc:
                record.status = "failed"
                record.error_message = exc.message
                return await self._persist(record)

        try:
            probed = await probe_bytes(video_bytes)
        except ProbeError as exc:
            record.status = "failed"
            record.error_message = exc.message
            return await self._persist(record)

        # Kling/http follow first-frame aspect + quality tier, not exact WxH.
        loose_geometry = record.model_provider in {"kling", "http"}
        assessment = assess_output(
            probed,
            requested_duration=record.requested_duration_seconds,
            requested_width=record.requested_width,
            requested_height=record.requested_height,
            requested_fps=float(record.requested_fps),
            min_duration=self.settings.hook_video_min_duration_seconds,
            max_duration=self.settings.hook_video_max_duration_seconds,
            duration_tolerance=self.settings.hook_video_duration_tolerance_seconds,
            strict_resolution=not loose_geometry,
        )
        _attach_measurements(record, probed)
        _extend(record.warnings, assessment.warnings)
        if not assessment.ok:
            record.status = "failed"
            record.output_key = None
            record.error_message = "; ".join(assessment.errors)
            logger.warning(
                "Hook video rejected job_id=%s error=%s",
                record.job_id,
                record.error_message,
            )
            return await self._persist(record)

        # Cover-crop to requested 9:16 (e.g. 1080x1920) so compose has no letterbox.
        target_w = record.requested_width
        target_h = record.requested_height
        if (
            loose_geometry
            and target_w
            and target_h
            and (probed.width, probed.height) != (target_w, target_h)
        ):
            src_w, src_h = probed.width, probed.height
            try:
                video_bytes = await normalize_video_cover(
                    video_bytes,
                    width=target_w,
                    height=target_h,
                    fps=max(12, int(round(record.requested_fps or probed.fps or 30))),
                    source_suffix=".mp4",
                    has_audio=False,
                )
                probed = await probe_bytes(video_bytes)
                _attach_measurements(record, probed)
                _extend(
                    record.warnings,
                    [f"hook_normalized_cover:{src_w}x{src_h}->{target_w}x{target_h}"],
                )
            except ProbeError as exc:
                _extend(record.warnings, [f"hook_normalize_skipped:{exc.message}"])

        extension = ".mov" if assessment.container == "mov" else ".mp4"
        record.output_key = await self.store.save_output(
            record.job_id,
            video_bytes,
            extension=extension,
        )
        record.status = "success"
        record.error_message = None
        logger.info(
            "Hook video ready job_id=%s duration=%s %sx%s fps=%s",
            record.job_id,
            record.duration_seconds,
            record.width,
            record.height,
            record.fps,
        )
        return await self._persist(record)

    async def _persist(self, record: JobRecord) -> HookVideoJobView:
        record.updated_at = _now()
        await self.store.save_job(record)
        return self._view(record)

    def _view(self, record: JobRecord) -> HookVideoJobView:
        url = None
        if record.status == "success" and record.output_key:
            url = self.storage.get_url(record.output_key)
        return HookVideoJobView(
            job_id=record.job_id,
            video_id=record.video_id,
            status=record.status,
            generated_video_url=url,
            duration_seconds=record.duration_seconds,
            width=record.width,
            height=record.height,
            fps=record.fps,
            model_provider=record.model_provider,
            model_name=record.model_name,
            warnings=list(record.warnings),
            error_message=record.error_message,
            comparison_id=record.comparison_id,
        )

    def _record(
        self,
        *,
        job_id: UUID,
        request: HookVideoGenerateRequest,
        provider_name: str,
        model_name: str,
        frame_key: str,
        width: int,
        height: int,
        warnings: list[str],
        comparison_id: UUID | None,
        created_at: str,
        status: str,
        provider_job_id: str | None = None,
        error_message: str | None = None,
    ) -> JobRecord:
        return JobRecord(
            job_id=job_id,
            video_id=request.video_id,
            status=status,  # type: ignore[arg-type]
            provider_job_id=provider_job_id,
            model_provider=provider_name,
            model_name=model_name,
            first_frame_url=request.first_frame_url,
            first_frame_key=frame_key,
            requested_duration_seconds=request.duration_seconds,
            requested_width=width,
            requested_height=height,
            requested_fps=request.fps,
            warnings=list(warnings),
            error_message=error_message,
            comparison_id=comparison_id,
            created_at=created_at,
            updated_at=created_at,
        )


def _extend(warnings: list[str], extra: list[str] | tuple[str, ...]) -> None:
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)


def _attach_measurements(record: JobRecord, probed) -> None:  # noqa: ANN001
    if probed.duration_seconds > 0:
        record.duration_seconds = round(probed.duration_seconds, 3)
    if probed.width > 0:
        record.width = probed.width
    if probed.height > 0:
        record.height = probed.height
    if probed.fps > 0:
        record.fps = round(probed.fps, 3)
