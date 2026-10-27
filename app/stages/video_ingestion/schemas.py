from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field


class IngestFromUrlRequest(BaseModel):
    url: AnyHttpUrl = Field(..., description="Temporary public URL to an MP4/MOV file")


class VideoMetadata(BaseModel):
    duration_seconds: float
    fps: float
    width: int
    height: int
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool


class VideoIngestionResponse(BaseModel):
    video_id: UUID
    video_url: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    has_audio: bool
    audio_url: str | None = None
    status: Literal["success"] = "success"
    video_codec: str | None = None
    audio_codec: str | None = None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
