"""API contracts for the hook video generator."""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

_RESOLUTION = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")
_ASPECTS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0}

MOTION_PLAN_DESCRIPTION = (
    "Motion plan object. Recognized keys: emotion, emotion_intensity, "
    "gaze_direction, facial_expression, camera_motion, motionPrompt "
    "(aliases motion_prompt, prompt), and motion_timeline items with "
    "start_seconds, end_seconds, head_motion, eye_motion, mouth_motion, "
    "body_motion, emotion. Unknown keys are ignored."
)


def parse_resolution(value: str) -> tuple[int, int]:
    match = _RESOLUTION.match(value)
    if not match:
        raise ValueError("output_resolution must look like 1080x1920")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64:
        raise ValueError("output_resolution must be at least 64x64")
    if width > 3840 or height > 3840:
        raise ValueError("output_resolution must be at most 3840 on each side")
    return width, height


def aspect_matches(aspect_ratio: str, width: int, height: int) -> bool:
    expected = _ASPECTS.get(aspect_ratio)
    if expected is None or height <= 0:
        return False
    return abs((width / height) - expected) <= 0.02


class _SourceFields(BaseModel):
    video_id: UUID
    first_frame_url: str = Field(..., min_length=1)
    motion_plan: dict[str, Any] = Field(
        default_factory=dict,
        description=MOTION_PLAN_DESCRIPTION,
    )

    @field_validator("first_frame_url")
    @classmethod
    def strip_first_frame_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("first_frame_url must not be empty")
        return cleaned


class _OutputFields(BaseModel):
    duration_seconds: float = Field(3, ge=2, le=4)
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    output_resolution: str = "1080x1920"
    fps: int = Field(30, ge=12, le=60)

    @field_validator("output_resolution")
    @classmethod
    def normalize_resolution(cls, value: str) -> str:
        width, height = parse_resolution(value)
        return f"{width}x{height}"

    @model_validator(mode="after")
    def resolution_matches_aspect(self) -> _OutputFields:
        width, height = parse_resolution(self.output_resolution)
        if width % 2 or height % 2:
            raise ValueError("output width and height must be even numbers")
        if not aspect_matches(self.aspect_ratio, width, height):
            raise ValueError(
                f"output_resolution {width}x{height} does not match aspect_ratio {self.aspect_ratio}"
            )
        return self


class HookVideoGenerateRequest(_SourceFields, _OutputFields):
    """Image-to-video request. model_provider and model_name select the adapter."""

    model_provider: str = Field(
        default="kling",
        description="Provider id from the registry (kling, stub, http).",
    )
    model_name: str = Field(
        default="kling-3.0",
        description=(
            "Model id for the selected provider. "
            "For kling: kling-3.0 (default) or kling-3.0-turbo."
        ),
    )

    @field_validator("model_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("model_provider must not be empty")
        return cleaned

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_name must not be empty")
        return cleaned


class ModelTarget(BaseModel):
    model_provider: str
    model_name: str

    @field_validator("model_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("model_provider must not be empty")
        return cleaned

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_name must not be empty")
        return cleaned


class HookVideoCompareRequest(_SourceFields, _OutputFields):
    """Same source and output spec, rendered by two or more models."""

    models: list[ModelTarget] = Field(..., min_length=2, max_length=4)


class HookVideoRerunRequest(BaseModel):
    """Optional overrides. Archived first frame is always reused."""

    model_provider: str | None = None
    model_name: str | None = None
    motion_prompt: str | None = Field(
        default=None,
        description="Replace the stored motionPrompt while keeping the archived frame.",
        max_length=8000,
    )
    motion_plan: dict[str, Any] | None = Field(
        default=None,
        description="Optional full motion_plan override. Takes precedence over motion_prompt.",
    )
    duration_seconds: float | None = Field(default=None, ge=2, le=4)

    @field_validator("model_provider")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("model_provider must not be empty")
        return cleaned

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_name must not be empty")
        return cleaned

    @field_validator("motion_prompt")
    @classmethod
    def strip_motion_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class HookVideoCallback(BaseModel):
    """Vendor completion notice. `provider_status: succeeded` still requires a valid file."""

    provider_status: str = Field(..., min_length=1)
    video_url: str | None = None
    error: str | None = None

    @field_validator("provider_status")
    @classmethod
    def strip_status(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("provider_status must not be empty")
        return cleaned

    @field_validator("video_url")
    @classmethod
    def strip_video_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class HookVideoAccepted(BaseModel):
    """Task accepted. Status is queued or processing, never success."""

    job_id: UUID
    video_id: UUID
    status: Literal["queued", "processing"]
    model_provider: str
    model_name: str
    warnings: list[str] = Field(default_factory=list)
    comparison_id: UUID | None = None


class HookVideoJobView(BaseModel):
    """Job state. `success` is present only after the output file passes validation."""

    job_id: UUID
    video_id: UUID
    status: Literal["queued", "processing", "success", "failed"]
    generated_video_url: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    model_provider: str
    model_name: str
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    comparison_id: UUID | None = None

    @model_validator(mode="after")
    def success_requires_validated_file(self) -> HookVideoJobView:
        if self.status == "success":
            if not self.generated_video_url:
                raise ValueError("success requires generated_video_url")
            if self.duration_seconds is None or self.duration_seconds <= 0:
                raise ValueError("success requires duration_seconds")
            if not self.width or not self.height:
                raise ValueError("success requires width and height")
            if self.fps is None or self.fps <= 0:
                raise ValueError("success requires fps")
        elif self.generated_video_url:
            raise ValueError("generated_video_url is set only when status is success")
        return self


class HookVideoCompareAccepted(BaseModel):
    comparison_id: UUID
    video_id: UUID
    jobs: list[HookVideoAccepted]


class HookVideoComparisonView(BaseModel):
    comparison_id: UUID
    video_id: UUID
    jobs: list[HookVideoJobView]


class StoredGenerationRequest(BaseModel):
    """Inputs kept so a job can be run again without resending the frame."""

    video_id: UUID
    first_frame_url: str
    first_frame_key: str
    motion_plan: dict[str, Any]
    duration_seconds: float
    aspect_ratio: str
    output_resolution: str
    width: int
    height: int
    fps: int
    model_provider: str
    model_name: str
    compiled_prompt: str
    negative_prompt: str
    identity_lock: str
    camera_lock: str


class JobRecord(BaseModel):
    job_id: UUID
    video_id: UUID
    status: Literal["queued", "processing", "success", "failed"]
    provider_job_id: str | None = None
    model_provider: str
    model_name: str
    first_frame_url: str
    first_frame_key: str
    output_key: str | None = None
    requested_duration_seconds: float
    requested_width: int
    requested_height: int
    requested_fps: int
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    comparison_id: UUID | None = None
    created_at: str
    updated_at: str


class ComparisonRecord(BaseModel):
    comparison_id: UUID
    video_id: UUID
    job_ids: list[UUID]


class ProviderListResponse(BaseModel):
    providers: list[str]


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
