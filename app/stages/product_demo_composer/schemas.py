"""API contracts for Product Demo Composer: [AI Hook] → [Product Demo]."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

_RESOLUTION = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")
_ASPECTS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0}

Transition = Literal["none", "fade", "crossfade"]
AudioMode = Literal["both", "hook", "demo", "none"]
OutputFormat = Literal["mp4", "mov"]
FitMode = Literal["blur", "cover", "contain"]
JobStatus = Literal["queued", "processing", "success", "failed"]


def parse_resolution(value: str) -> tuple[int, int]:
    match = _RESOLUTION.match(value)
    if not match:
        raise ValueError("resolution must look like 1080x1920")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64:
        raise ValueError("resolution must be at least 64x64")
    if width > 3840 or height > 3840:
        raise ValueError("resolution must be at most 3840 on each side")
    return width, height


def aspect_matches(aspect_ratio: str, width: int, height: int) -> bool:
    expected = _ASPECTS.get(aspect_ratio)
    if expected is None or height <= 0:
        return False
    return abs((width / height) - expected) <= 0.02


class VideoAssetRef(BaseModel):
    """Reference to a stored or remote video artifact (never a local path)."""

    asset_id: UUID
    video_url: str = Field(..., min_length=1)

    @field_validator("video_url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("video_url must not be empty")
        if any(token in cleaned for token in ("\x00", "\n", "\r")):
            raise ValueError("video_url contains invalid characters")
        return cleaned


class ComposeSettings(BaseModel):
    """Output + transition + audio policy. Defaults preserve hook cut mechanics."""

    output_format: OutputFormat = "mp4"
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    resolution: str = "1080x1920"
    fps: int = Field(30, ge=12, le=60)
    fit_mode: FitMode = Field(
        "blur",
        description=(
            "blur = fit full frame, fill empty areas with a blurred copy of the "
            "same clip (Reels/TikTok style, no black bars). "
            "cover = fill 9:16 by cropping edges. "
            "contain = letterbox/pillarbox with black bars."
        ),
    )
    transition: Transition = "none"
    transition_duration_seconds: float = Field(0, ge=0, le=5)
    audio_mode: AudioMode = Field(
        "both",
        description=(
            "both = keep each segment's audio; hook/demo = keep only that "
            "segment; none = mute both. No music overlay in this service."
        ),
    )

    @field_validator("resolution")
    @classmethod
    def normalize_resolution(cls, value: str) -> str:
        width, height = parse_resolution(value)
        return f"{width}x{height}"

    @model_validator(mode="after")
    def validate_geometry_and_transition(self) -> ComposeSettings:
        width, height = parse_resolution(self.resolution)
        if width % 2 or height % 2:
            raise ValueError("resolution width and height must be even numbers")
        if not aspect_matches(self.aspect_ratio, width, height):
            raise ValueError(
                f"resolution {width}x{height} does not match aspect_ratio {self.aspect_ratio}"
            )
        if self.transition == "none" and self.transition_duration_seconds > 0:
            raise ValueError(
                "transition_duration_seconds must be 0 when transition is none"
            )
        if self.transition != "none" and self.transition_duration_seconds <= 0:
            raise ValueError(
                "transition_duration_seconds must be > 0 for fade/crossfade"
            )
        return self


class ComposeRequest(BaseModel):
    """Logical compose contract (hook URL + demo asset).

    The HTTP endpoint accepts demo as a multipart file upload in Swagger;
    after archive the demo is stored as VideoAssetRef (asset_id + video_url).
    """

    project_id: UUID
    hook_video: VideoAssetRef
    demo_video: VideoAssetRef
    settings: ComposeSettings = Field(default_factory=ComposeSettings)


class ComposeRerunRequest(BaseModel):
    """Re-run composition from archived inputs with optional settings override."""

    settings: ComposeSettings | None = None


class TimelineSegment(BaseModel):
    type: Literal["hook", "product_demo"]
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> TimelineSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class ComposeAccepted(BaseModel):
    """Job accepted. Poll GET /jobs/{job_id} for the composed asset."""

    project_id: UUID
    job_id: UUID
    status: Literal["queued", "processing"]
    warnings: list[str] = Field(default_factory=list)


class ComposeJobView(BaseModel):
    """Job state. Output fields are present only when status is success."""

    project_id: UUID
    job_id: UUID
    asset_id: UUID | None = None
    video_url: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    segments: list[TimelineSegment] = Field(default_factory=list)
    status: JobStatus
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def success_requires_output(self) -> ComposeJobView:
        if self.status == "success":
            if not self.asset_id or not self.video_url:
                raise ValueError("success requires asset_id and video_url")
            if self.duration_seconds is None or self.duration_seconds <= 0:
                raise ValueError("success requires duration_seconds")
            if not self.width or not self.height:
                raise ValueError("success requires width and height")
            if self.fps is None or self.fps <= 0:
                raise ValueError("success requires fps")
            if len(self.segments) != 2:
                raise ValueError("success requires hook and product_demo segments")
        elif self.video_url:
            raise ValueError("video_url is set only when status is success")
        return self


class StoredComposeRequest(BaseModel):
    """Persisted inputs so composition can be re-run without re-uploading."""

    project_id: UUID
    hook_asset_id: UUID
    demo_asset_id: UUID
    hook_video_url: str
    demo_video_url: str
    hook_video_key: str
    demo_video_key: str
    settings: ComposeSettings


class JobRecord(BaseModel):
    job_id: UUID
    project_id: UUID
    status: JobStatus
    hook_asset_id: UUID
    demo_asset_id: UUID
    hook_video_key: str
    demo_video_key: str
    settings: ComposeSettings
    asset_id: UUID | None = None
    output_key: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    segments: list[TimelineSegment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    parent_job_id: UUID | None = None
    created_at: str
    updated_at: str


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
