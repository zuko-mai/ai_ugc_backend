"""Local image-to-video provider for tests and dry-runs.

`submit` only stores the task. The still-frame video is rendered on a later poll
so a queued response is never treated as a finished video. Appearance and
framing stay identical to the first frame because the file is that frame held
for the requested duration. Real motion requires another registered provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from app.stages.hook_video_generator.exceptions import ProviderApiError
from app.stages.hook_video_generator.providers.base import (
    VideoGenerationProvider,
    VideoProviderPoll,
    VideoProviderRequest,
    VideoProviderSubmission,
)
from app.stages.hook_video_generator.render import RenderError, render_still_video

STUB_QUEUED_WARNING = (
    "stub_provider: task queued; the video is produced on a later poll"
)
STUB_HOLD_WARNING = (
    "stub_provider: first frame held for the requested duration; "
    "no generative motion was applied"
)

_SUFFIX_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


@dataclass
class _StubJob:
    request: VideoProviderRequest
    polls_until_ready: int
    polls: int = 0
    result: VideoProviderPoll | None = None


_JOBS: dict[str, _StubJob] = {}
_LOCK = asyncio.Lock()


def reset_stub_jobs() -> None:
    _JOBS.clear()


class StubVideoProvider(VideoGenerationProvider):
    name = "stub"

    def __init__(self, *, polls_until_ready: int = 2) -> None:
        if polls_until_ready < 1:
            raise ValueError("polls_until_ready must be at least 1")
        self.polls_until_ready = polls_until_ready

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        provider_job_id = str(uuid4())
        async with _LOCK:
            _JOBS[provider_job_id] = _StubJob(
                request=request,
                polls_until_ready=self.polls_until_ready,
            )
        return VideoProviderSubmission(
            provider_job_id=provider_job_id,
            status="queued",
            provider_warnings=(STUB_QUEUED_WARNING,),
        )

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        async with _LOCK:
            job = _JOBS.get(provider_job_id)
            if job is None:
                raise ProviderApiError(
                    "Stub video task was not found",
                    details={"provider_job_id": provider_job_id},
                )
            if job.result is not None:
                return job.result
            job.polls += 1
            if job.polls < job.polls_until_ready:
                return VideoProviderPoll(
                    provider_job_id=provider_job_id,
                    status="processing",
                )
            request = job.request

        try:
            video = await render_still_video(
                request.first_frame.data,
                width=request.width,
                height=request.height,
                fps=request.fps,
                duration_seconds=request.duration_seconds,
                image_suffix=_SUFFIX_BY_MIME.get(request.first_frame.mime_type, ".jpg"),
            )
        except RenderError as exc:
            raise ProviderApiError(str(exc)) from exc

        result = VideoProviderPoll(
            provider_job_id=provider_job_id,
            status="succeeded",
            video_bytes=video,
            mime_type="video/mp4",
            provider_warnings=(STUB_HOLD_WARNING,),
        )
        async with _LOCK:
            current = _JOBS.get(provider_job_id)
            if current is not None:
                current.result = result
        return result
