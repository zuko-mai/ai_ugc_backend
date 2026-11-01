"""API contracts for Music Overlay.

`music_start_offset_seconds` and `video_start_seconds` are independent:
the first seeks inside the music file, the second places that excerpt on
the output video timeline.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

MusicEndBehavior = Literal["silence", "loop", "trim"]
OutputFormat = Literal["mp4", "mov"]
JobStatus = Literal["queued", "processing", "success", "failed"]


def _clean_url(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("url must not be empty")
    if any(token in cleaned for token in ("\x00", "\n", "\r")):
        raise ValueError("url contains invalid characters")
    return cleaned


class VideoAssetRef(BaseModel):
    """Reference to a finished video. This stage does not care how it was made."""

    asset_id: UUID
    video_url: str = Field(..., min_length=1)

    @field_validator("video_url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        return _clean_url(value)


class MusicAssetRef(BaseModel):
    """Reference to a music file (MP3, WAV, AAC, M4A, and other common types)."""

    asset_id: UUID
    audio_url: str = Field(..., min_length=1)

    @field_validator("audio_url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        return _clean_url(value)


class OverlaySettings(BaseModel):
    """How the music bed is placed, leveled, and faded on the video."""

    music_start_offset_seconds: float = Field(
        0,
        ge=0,
        description="Seek position inside the source music file, in seconds.",
    )
    video_start_seconds: float = Field(
        0,
        ge=0,
        description="When the music becomes audible on the output video, in seconds.",
    )
    music_volume: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Linear music gain. 0.3 is 30 percent.",
    )
    original_audio_enabled: bool = True
    original_audio_volume: float = Field(1.0, ge=0, le=1)
    fade_in_seconds: float = Field(0, ge=0, le=120)
    fade_out_seconds: float = Field(0, ge=0, le=120)
    loop_music: bool = False
    music_end_behavior: MusicEndBehavior = Field(
        "silence",
        description=(
            "What to do when the excerpt ends before the video: "
            "silence or trim keep the video and do not repeat the music; "
            "loop repeats the excerpt from music_start_offset_seconds."
        ),
    )
    output_format: OutputFormat = "mp4"

    @model_validator(mode="after")
    def reconcile_loop_flags(self) -> OverlaySettings:
        """Keep loop_music and music_end_behavior consistent.

        `music_end_behavior=loop` wins over a default `loop_music=false`.
        `loop_music=true` with the default silence behavior selects loop.
        Combining `loop_music=true` with `trim` is rejected.
        """
        if self.loop_music and self.music_end_behavior == "trim":
            raise ValueError(
                "loop_music=true conflicts with music_end_behavior='trim'"
            )
        if self.loop_music and self.music_end_behavior != "loop":
            self.music_end_behavior = "loop"
        if self.music_end_behavior == "loop":
            self.loop_music = True
        else:
            self.loop_music = False
        return self


class OverlayRequest(BaseModel):
    """JSON body for POST /v1/music-overlay/overlay."""

    project_id: UUID
    video: VideoAssetRef
    music: MusicAssetRef
    settings: OverlaySettings = Field(default_factory=OverlaySettings)


class OverlayRerunRequest(BaseModel):
    """Re-run from archived inputs. Omit settings to repeat the same mix."""

    settings: OverlaySettings | None = None


class AudioReport(BaseModel):
    """Applied mix parameters echoed on a successful job."""

    music_start_offset_seconds: float = Field(..., ge=0)
    video_start_seconds: float = Field(..., ge=0)
    music_volume: float = Field(..., ge=0, le=1)
    original_audio_enabled: bool
    fade_in_seconds: float = Field(..., ge=0)
    fade_out_seconds: float = Field(..., ge=0)


class OverlayAccepted(BaseModel):
    """Job accepted. Poll GET /jobs/{job_id} for the mixed asset."""

    project_id: UUID
    job_id: UUID
    status: Literal["queued", "processing"]
    warnings: list[str] = Field(default_factory=list)


class OverlayJobView(BaseModel):
    """Job state. Asset fields are present only when status is success."""

    project_id: UUID
    job_id: UUID
    asset_id: UUID | None = None
    video_url: str | None = None
    duration_seconds: float | None = None
    audio: AudioReport | None = None
    status: JobStatus
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def success_requires_output(self) -> OverlayJobView:
        if self.status == "success":
            if not self.asset_id or not self.video_url:
                raise ValueError("success requires asset_id and video_url")
            if self.duration_seconds is None or self.duration_seconds <= 0:
                raise ValueError("success requires duration_seconds")
            if self.audio is None:
                raise ValueError("success requires audio")
        elif self.video_url or self.asset_id or self.audio is not None:
            raise ValueError("asset fields are set only when status is success")
        return self


class StoredOverlayRequest(BaseModel):
    """Persisted inputs so a mix can be re-run without re-uploading."""

    project_id: UUID
    video_asset_id: UUID
    music_asset_id: UUID
    video_url: str
    audio_url: str
    video_key: str
    music_key: str
    settings: OverlaySettings


class JobRecord(BaseModel):
    job_id: UUID
    project_id: UUID
    status: JobStatus
    video_asset_id: UUID
    music_asset_id: UUID
    video_key: str
    music_key: str
    settings: OverlaySettings
    asset_id: UUID | None = None
    output_key: str | None = None
    duration_seconds: float | None = None
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
