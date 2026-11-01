from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.stages.face_replacement.examples import (
    TEST_SELECTED_FRAME_URL,
    TEST_VIDEO_ID,
    TEST_VIDEO_URL,
)


class FaceReplacementRequest(BaseModel):
    """Form fields for face replacement (face image is uploaded separately).

    Required:
      - video_id
      - selected_frame_url
      - face_reference (multipart file — not in this model; see API)

    Optional (defaults applied when omitted):
      - video_url
      - remove_overlay_text (true)
      - preserve_original_composition (true)
      - preserve_pose (true)
      - preserve_lighting (true)
      - model_provider (gemini)
      - model_name (gemini-3-pro-image / Nano Banana Pro; alt: gemini-3.1-flash-image / Nano Banana 2)
    """

    video_id: UUID = Field(
        ...,
        description="REQUIRED. Video id from ingestion / earlier pipeline stages.",
        json_schema_extra={"required_field": True, "examples": [TEST_VIDEO_ID]},
        examples=[TEST_VIDEO_ID],
    )
    video_url: str | None = Field(
        default=None,
        description=(
            "OPTIONAL. Source video URL for pipeline/context (not used for the swap itself)."
        ),
        json_schema_extra={"required_field": False, "examples": [TEST_VIDEO_URL]},
        examples=[TEST_VIDEO_URL],
    )
    selected_frame_url: str = Field(
        ...,
        min_length=1,
        description="REQUIRED. URL of the selected UGC frame (master scene / FIRST image).",
        json_schema_extra={"required_field": True, "examples": [TEST_SELECTED_FRAME_URL]},
        examples=[TEST_SELECTED_FRAME_URL],
    )
    remove_overlay_text: bool = Field(
        True,
        description="OPTIONAL. Remove burned-in captions/stickers. Default: true.",
        json_schema_extra={"required_field": False},
    )
    preserve_original_composition: bool = Field(
        True,
        description="OPTIONAL. Keep framing/composition of the base frame. Default: true.",
        json_schema_extra={"required_field": False},
    )
    preserve_pose: bool = Field(
        True,
        description="OPTIONAL. Keep head/body pose from the base frame. Default: true.",
        json_schema_extra={"required_field": False},
    )
    preserve_lighting: bool = Field(
        True,
        description="OPTIONAL. Match lighting/shadows of the base frame. Default: true.",
        json_schema_extra={"required_field": False},
    )
    model_provider: str = Field(
        default="gemini",
        description="OPTIONAL. Provider id (gemini, stub). Default: gemini.",
        json_schema_extra={"required_field": False},
    )
    model_name: str = Field(
        default="gemini-3-pro-image",
        description=(
            "OPTIONAL. Image model id. "
            "Default: gemini-3-pro-image (Nano Banana Pro). "
            "Secondary: gemini-3.1-flash-image (Nano Banana 2). "
            "Aliases: nano-banana-pro, nano-banana-2."
        ),
        json_schema_extra={
            "required_field": False,
            "examples": ["gemini-3-pro-image", "gemini-3.1-flash-image"],
        },
        examples=["gemini-3-pro-image", "gemini-3.1-flash-image"],
    )
    face_reference_url: str | None = Field(
        default=None,
        description=(
            "OPTIONAL. Reuse a previously stored face reference by URL "
            "(/media/uploads/... or /media/face_replacement/...). "
            "Provide this OR multipart face_reference."
        ),
    )
    user_prompt: str | None = Field(
        default=None,
        description="OPTIONAL. Extra instructions appended to the face-replacement prompt.",
        max_length=4000,
    )

    @field_validator("selected_frame_url", "video_url", "face_reference_url")
    @classmethod
    def strip_urls(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("URL must not be empty")
        return cleaned

    @field_validator("model_provider", "model_name")
    @classmethod
    def strip_model_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("user_prompt")
    @classmethod
    def strip_user_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class GenerationUsage(BaseModel):
    """Tokens, estimated spend, and wall-clock timing for one generate call."""

    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    currency: str = "USD"
    processing_time_seconds: float = Field(
        ...,
        ge=0,
        description="End-to-end wall time for the face-replacement request",
    )
    provider_latency_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Time spent inside the model provider call only",
    )
    pricing_note: str | None = None


class FaceReplacementResponse(BaseModel):
    video_id: UUID
    generated_frame_url: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    model_provider: str
    model_name: str
    text_removal_requested: bool
    status: Literal["success"] = "success"
    warnings: list[str] = Field(default_factory=list)
    generation_id: UUID | None = Field(
        default=None,
        description="Id of this generation; supports regeneration history",
    )
    usage: GenerationUsage = Field(
        ...,
        description="Token counts, estimated USD cost, and processing time",
    )


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None


class GenerationRecord(BaseModel):
    """Persisted metadata for one face-replacement generation."""

    generation_id: UUID
    video_id: UUID
    video_url: str | None = None
    selected_frame_url: str
    face_reference_filename: str | None = None
    face_reference_key: str | None = None
    face_reference_url: str | None = None
    generated_frame_key: str
    width: int
    height: int
    model_provider: str
    model_name: str
    text_removal_requested: bool
    preserve_original_composition: bool
    preserve_pose: bool
    preserve_lighting: bool
    user_prompt: str | None = None
    compiled_prompt: str | None = None
    warnings: list[str] = Field(default_factory=list)
    status: Literal["success"] = "success"
    usage: GenerationUsage | None = None


class FaceRerunRequest(BaseModel):
    """Regenerate from archived inputs. Optional overrides only."""

    user_prompt: str | None = Field(default=None, max_length=4000)
    remove_overlay_text: bool | None = None
    preserve_original_composition: bool | None = None
    preserve_pose: bool | None = None
    preserve_lighting: bool | None = None
    model_provider: str | None = None
    model_name: str | None = None

    @field_validator("user_prompt")
    @classmethod
    def strip_user_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("model_provider", "model_name")
    @classmethod
    def strip_optional_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned
