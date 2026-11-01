"""Video provider integration interface.

Business logic depends only on this module. A new vendor is a new subclass
registered under a `model_provider` id. `model_name` is passed through unchanged.

Lifecycle:
1. `submit` queues an image-to-video task and returns a provider job id.
2. `poll` reports queued, processing, succeeded, or failed.
3. A succeeded poll includes video bytes or a URL. The service downloads the
   file and checks duration, container, and resolution before status `success`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from app.stages.hook_video_generator.instructions import MotionInstructions

RemoteStatus = Literal["queued", "processing", "succeeded", "failed"]

_REMOTE_STATUS: dict[str, RemoteStatus] = {
    "queued": "queued",
    "pending": "queued",
    "submitted": "queued",
    "accepted": "queued",
    "processing": "processing",
    "running": "processing",
    "in_progress": "processing",
    "in-progress": "processing",
    "succeeded": "succeeded",
    "success": "succeeded",
    "completed": "succeeded",
    "complete": "succeeded",
    "done": "succeeded",
    "failed": "failed",
    "failure": "failed",
    "error": "failed",
    "cancelled": "failed",
    "canceled": "failed",
}


def normalize_remote_status(raw: str) -> RemoteStatus:
    """Map a vendor status string onto the shared task states."""
    key = raw.strip().lower().replace(" ", "_")
    try:
        return _REMOTE_STATUS[key]
    except KeyError as exc:
        raise ValueError(raw) from exc


@dataclass(frozen=True)
class ProviderImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    source_url: str


@dataclass(frozen=True)
class VideoProviderRequest:
    first_frame: ProviderImage
    instructions: MotionInstructions
    duration_seconds: float
    aspect_ratio: str
    width: int
    height: int
    fps: int
    model_name: str
    timeout_seconds: float


@dataclass(frozen=True)
class VideoProviderSubmission:
    """Acknowledgement that a task was queued. This is not a finished video."""

    provider_job_id: str
    status: Literal["queued", "processing"]
    provider_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoProviderPoll:
    provider_job_id: str
    status: RemoteStatus
    video_bytes: bytes | None = None
    video_url: str | None = None
    mime_type: str | None = None
    error_message: str | None = None
    provider_warnings: tuple[str, ...] = ()


class VideoGenerationProvider(ABC):
    """Adapter boundary between the hook video stage and a model API."""

    name: str

    @abstractmethod
    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        """Queue image-to-video generation and return a task id."""

    @abstractmethod
    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        """Return the current task state. Include a file only when the vendor finished."""
