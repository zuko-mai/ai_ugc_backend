from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class HookInterval(BaseModel):
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "HookInterval":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds > 15:
            raise ValueError("hook interval must be at most 15 seconds")
        return self


class VideoRefMixin(BaseModel):
    """Accept either stored video_id or a video_url (or both)."""

    video_id: UUID | None = Field(
        default=None,
        description="Id from ingestion; resolves originals/{id}/source.* from storage",
    )
    video_url: str | None = Field(
        default=None,
        min_length=1,
        description="Local /media/... URL or remote direct file URL",
    )

    @model_validator(mode="after")
    def require_video_id_or_url(self) -> "VideoRefMixin":
        if self.video_id is None and not self.video_url:
            raise ValueError("Provide video_id or video_url (at least one)")
        return self


class CandidatesRequest(VideoRefMixin):
    hook_start_seconds: float = Field(0, ge=0)
    hook_end_seconds: float = Field(4, gt=0)
    candidate_count: int = Field(5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_interval(self) -> "CandidatesRequest":
        HookInterval(
            start_seconds=self.hook_start_seconds,
            end_seconds=self.hook_end_seconds,
        )
        return self


class MotionPlanRequest(VideoRefMixin):
    # User-facing trim window for Gemini / Kling (defaults: from 0s, length 3s).
    start_seconds: float = Field(
        0,
        ge=0,
        description="Where to start the clip in the source video (seconds)",
    )
    duration_seconds: float = Field(
        3,
        gt=0,
        le=15,
        description="How long the clip lasts (seconds)",
    )

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds

    @model_validator(mode="after")
    def validate_window(self) -> "MotionPlanRequest":
        HookInterval(
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        return self


class ReanalyzeRequest(BaseModel):
    """Re-run motion analysis after the user selects a candidate frame.

    Uses cached frames/landmarks — does not re-extract the source video.
    """

    video_id: UUID
    selected_frame_id: UUID
    start_seconds: float = Field(0, ge=0)
    duration_seconds: float = Field(3, gt=0, le=15)

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds

    @model_validator(mode="after")
    def validate_window(self) -> "ReanalyzeRequest":
        HookInterval(
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        return self


class FrameCandidate(BaseModel):
    frame_id: UUID
    timestamp_seconds: float
    image_url: str
    quality_score: float = Field(..., ge=0, le=1)
    reason: str


class CandidatesResponse(BaseModel):
    video_id: UUID
    candidates: list[FrameCandidate]
    analysis_key: str = Field(
        ...,
        description="Cache key for this interval; used by reanalyze without re-extraction",
    )


class MotionTimelineItem(BaseModel):
    start_seconds: float
    end_seconds: float
    head_motion: str
    eye_motion: str
    mouth_motion: str
    body_motion: str
    emotion: str


class GeminiUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    currency: str = "USD"
    pricing_note: str | None = None


class MotionPlanResponse(BaseModel):
    video_id: UUID
    interval: HookInterval
    emotion: str
    emotion_intensity: float = Field(..., ge=0, le=1)
    gaze_direction: str
    facial_expression: str
    motion_timeline: list[MotionTimelineItem]
    audio_context: str | None = None
    confidence: float = Field(..., ge=0, le=1)
    selected_frame_id: UUID | None = None
    analysis_key: str | None = None
    motionPrompt: str | None = Field(
        default=None,
        description="Natural-language Kling 3.0 motion prompt from Gemini over the trimmed clip",
    )
    trimmed_video_url: str | None = Field(
        default=None,
        description="URL of the trimmed clip sent to Gemini (for manual review)",
    )
    gemini_usage: GeminiUsage | None = Field(
        default=None,
        description="Token counts and estimated USD cost for the Gemini motionPrompt call",
    )


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None


class FaceObservation(BaseModel):
    """Internal per-frame observation — only measurable signals."""

    timestamp_seconds: float
    face_detected: bool
    face_bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    face_center: tuple[float, float] | None = None
    face_area_ratio: float = 0.0
    sharpness: float = 0.0
    edge_clipped: bool = False
    mouth_openness: float | None = None
    gaze_offset_x: float | None = None
    gaze_offset_y: float | None = None

    @field_validator("face_bbox", mode="before")
    @classmethod
    def tupleize_bbox(cls, value):  # noqa: ANN001
        if value is None:
            return None
        return tuple(value)
