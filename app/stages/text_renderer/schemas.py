"""API contracts for the deterministic text renderer."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

_COLOR = re.compile(r"^#([0-9A-Fa-f]{6})$")
_FONT_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$")
_MAX_TEXT_CHARS = 2000
_MAX_SEGMENTS = 40


class TextStyle(BaseModel):
    """UGC caption style. Defaults match typical TikTok/Reels hook text."""

    font_family: str = "Arial"
    font_size: int = Field(
        72,
        ge=12,
        le=200,
        description="Pixel size in frame coordinates. ~64–84 looks close to UGC refs on 1080p.",
    )
    font_weight: Literal["regular", "bold", "black"] = Field(
        "bold",
        description="regular / bold / black (heavy). Prefer bold or black for hook captions.",
    )
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: float = Field(3.5, ge=0, le=16)
    padding_top: float = Field(
        72,
        ge=0,
        le=95,
        description=(
            "Vertical offset of the text block from the top of the frame, in percent "
            "(0 = flush top, ~70–80 ≈ lower third / UGC hook)."
        ),
    )
    text_align: Literal["left", "center", "right"] = "center"
    max_width_ratio: float = Field(
        0.88,
        ge=0.4,
        le=0.98,
        description="Fraction of frame width used for wrapping.",
    )
    line_spacing: float = Field(1.12, ge=1.0, le=2.0)

    @field_validator("font_family")
    @classmethod
    def validate_font_family(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not _FONT_FAMILY.match(cleaned):
            raise ValueError(
                "font_family must be a short alphanumeric name "
                "(letters, digits, spaces, ._-)"
            )
        return cleaned

    @field_validator("font_color", "outline_color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        cleaned = value.strip()
        if not _COLOR.match(cleaned):
            raise ValueError("color must be a hex value like #FFFFFF")
        return cleaned.upper()


class TextSegment(BaseModel):
    start_seconds: float = Field(..., ge=0)
    end_seconds: float | None = Field(
        default=None,
        description=(
            "Segment end time. Omit or set null to keep text until the end of the video, "
            "regardless of duration."
        ),
    )
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def preserve_text(cls, value: str) -> str:
        # Exact user text: strip only trailing outer whitespace from the field,
        # never rewrite wording. Reject NUL / control codes that break burn-in.
        if "\x00" in value:
            raise ValueError("text must not contain NUL bytes")
        for ch in value:
            code = ord(ch)
            if code < 32 and ch not in {"\n", "\r", "\t"}:
                raise ValueError("text contains unsupported control characters")
        cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
        if not cleaned.strip():
            raise ValueError("text must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_interval(self) -> TextSegment:
        if self.end_seconds is None:
            return self
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds > 120:
            raise ValueError("segment duration must be at most 120 seconds")
        return self


class TextRenderRequest(BaseModel):
    """Burn exact user text onto a generated video (or export without overlay)."""

    video_id: UUID
    generated_video_url: str = Field(..., min_length=1)
    text: str | None = Field(
        default=None,
        max_length=_MAX_TEXT_CHARS,
        description="Fallback full-clip text when segments are omitted.",
    )
    language: str = Field(default="en", min_length=2, max_length=16)
    style: TextStyle = Field(default_factory=TextStyle)
    segments: list[TextSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)
    output_format: Literal["mp4", "mov"] = "mp4"
    skip_text: bool = Field(
        False,
        description="If true, export a copy without burning any text overlay.",
    )

    @field_validator("generated_video_url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("generated_video_url must not be empty")
        if any(token in cleaned for token in ("\x00", "\n", "\r")):
            raise ValueError("generated_video_url contains invalid characters")
        return cleaned

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        cleaned = value.strip().lower().replace("_", "-")
        if not cleaned or not re.match(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$", cleaned):
            raise ValueError("language must look like en, ru, or en-US")
        return cleaned

    @field_validator("text")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("text must not contain NUL bytes")
        for ch in value:
            code = ord(ch)
            if code < 32 and ch not in {"\n", "\r", "\t"}:
                raise ValueError("text contains unsupported control characters")
        cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
        if not cleaned.strip():
            return None
        return cleaned

    @model_validator(mode="after")
    def require_text_or_skip(self) -> TextRenderRequest:
        if self.skip_text:
            return self
        if self.segments:
            return self
        if self.text:
            return self
        raise ValueError(
            "provide text and/or segments, or set skip_text=true to export without overlay"
        )


class TextPreviewRequest(TextRenderRequest):
    """Same inputs as render, plus an optional timestamp for a still preview."""

    preview_at_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Timestamp for the still preview. Defaults to the first segment start.",
    )


class TextRerenderRequest(BaseModel):
    """Re-apply text/style onto the archived source video without regenerating it."""

    text: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    style: TextStyle | None = None
    segments: list[TextSegment] | None = Field(default=None, max_length=_MAX_SEGMENTS)
    output_format: Literal["mp4", "mov"] | None = None
    skip_text: bool | None = None

    @field_validator("text")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return TextRenderRequest.validate_optional_text(value)

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return TextRenderRequest.normalize_language(value)


class TextRenderResponse(BaseModel):
    video_id: UUID
    render_id: UUID
    asset_id: UUID | None = Field(
        default=None,
        description="Same as render_id; stable id for product-demo-composer hook_asset_id.",
    )
    rendered_video_url: str
    duration_seconds: float
    width: int
    height: int
    format: Literal["mp4", "mov"]
    text_segments_rendered: int
    status: Literal["success"] = "success"
    preview_image_url: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_asset_id(self) -> TextRenderResponse:
        if self.asset_id is None:
            self.asset_id = self.render_id
        return self


class TextPreviewResponse(BaseModel):
    video_id: UUID
    render_id: UUID | None = None
    preview_image_url: str
    width: int
    height: int
    preview_at_seconds: float
    text_segments_rendered: int
    status: Literal["success"] = "success"
    warnings: list[str] = Field(default_factory=list)


class StoredRenderRequest(BaseModel):
    """Persisted inputs so text can be re-burned without regenerating the video."""

    video_id: UUID
    generated_video_url: str
    source_video_key: str
    text: str | None = None
    language: str = "en"
    style: TextStyle
    segments: list[TextSegment] = Field(default_factory=list)
    output_format: Literal["mp4", "mov"] = "mp4"
    skip_text: bool = False


class RenderRecord(BaseModel):
    render_id: UUID
    video_id: UUID
    source_video_key: str
    output_key: str
    preview_key: str | None = None
    duration_seconds: float
    width: int
    height: int
    format: Literal["mp4", "mov"]
    text_segments_rendered: int
    status: Literal["success"] = "success"
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    parent_render_id: UUID | None = None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
