from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

HookGoal = Literal[
    "curiosity",
    "urgency",
    "pain",
    "benefit",
    "social_proof",
    "question",
    "contrast",
]
ClaimsReview = Literal["pass", "fail", "review"]
TextSource = Literal["on_screen", "spoken", "manual", "composed"]


class ExtractionInterval(BaseModel):
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "ExtractionInterval":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds > 30:
            raise ValueError("extraction interval must be at most 30 seconds")
        return self


class ExtractionRequest(BaseModel):
    video_id: UUID | None = Field(
        default=None,
        description="Id from ingestion; resolves originals/{id}/source.* from storage",
    )
    video_url: str | None = Field(
        default=None,
        min_length=1,
        description="Local /media/... URL or remote direct file URL",
    )
    start_seconds: float = Field(0, ge=0)
    end_seconds: float = Field(4, gt=0)
    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        min_length=1,
        description="Preferred languages for OCR/ASR (e.g. en, ru)",
    )

    @model_validator(mode="after")
    def validate_request(self) -> "ExtractionRequest":
        if self.video_id is None and not self.video_url:
            raise ValueError("Provide video_id or video_url (at least one)")
        ExtractionInterval(
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        return self


class OnScreenTextFragment(BaseModel):
    text: str
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., ge=0)
    bbox: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Pixel bbox [x, y, width, height]",
    )
    confidence: float = Field(..., ge=0, le=1)
    language: str | None = None
    source: Literal["on_screen"] = "on_screen"


class SpokenTextFragment(BaseModel):
    text: str
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., ge=0)
    language: str
    confidence: float = Field(..., ge=0, le=1)
    source: Literal["spoken"] = "spoken"


class ExtractionResponse(BaseModel):
    video_id: UUID
    extraction_id: UUID
    start_seconds: float
    end_seconds: float
    on_screen_text: list[OnScreenTextFragment]
    spoken_text: list[SpokenTextFragment]
    languages: list[str]


class AdaptationRequest(BaseModel):
    original_text: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Source hook text. Optional when extraction_id is provided — "
            "then text is composed from the extraction session."
        ),
    )
    product_name: str = Field(..., min_length=1)
    product_description: str = Field(..., min_length=1)
    target_audience: str = Field(..., min_length=1)
    language: str = Field(..., min_length=2)
    hook_goal: HookGoal = "curiosity"
    variants_count: int = Field(5, ge=1, le=10)
    # Manual override always wins over auto-recognized / original_text.
    manual_text: str | None = Field(
        default=None,
        description="User-edited text; takes priority over original_text",
    )
    skip_adaptation: bool = Field(
        default=False,
        description="Keep original (or manual) text without generating variants",
    )
    extraction_id: UUID | None = Field(
        default=None,
        description=(
            "Prior extraction session. When original_text is omitted, "
            "compose_original_from_extraction() fills it."
        ),
    )
    preferred_source: TextSource | None = Field(
        default=None,
        description="Preferred extraction source when composing from extraction_id",
    )

    @model_validator(mode="after")
    def require_text_or_extraction(self) -> "AdaptationRequest":
        if self.manual_text and self.manual_text.strip():
            return self
        if self.original_text and self.original_text.strip():
            return self
        if self.extraction_id is not None:
            return self
        raise ValueError("Provide original_text, manual_text, or extraction_id")


class HookVariant(BaseModel):
    variant_id: UUID
    hook_text: str
    hook_type: HookGoal
    structure: str
    product_alignment: float = Field(..., ge=0, le=1)
    claims_review: ClaimsReview


class AdaptationResponse(BaseModel):
    adaptation_id: UUID
    original_text: str
    variants: list[HookVariant]
    selected_text: str | None = None
    skip_adaptation: bool = False
    preferred_source: TextSource | None = None
    extraction_id: UUID | None = None


class SelectRequest(BaseModel):
    """Finalize hook text. Priority: manual_text > variant_id > use_original."""

    adaptation_id: UUID
    manual_text: str | None = Field(
        default=None,
        description="Highest priority — user-edited final text",
    )
    variant_id: UUID | None = None
    use_original: bool = False

    @model_validator(mode="after")
    def require_selection(self) -> "SelectRequest":
        if self.manual_text is None and self.variant_id is None and not self.use_original:
            raise ValueError(
                "Provide manual_text, variant_id, or use_original=true"
            )
        return self


class SelectResponse(BaseModel):
    adaptation_id: UUID
    original_text: str
    variants: list[HookVariant]
    selected_text: str
    selection_mode: Literal["manual", "variant", "original"]


class SessionResponse(BaseModel):
    adaptation_id: UUID
    original_text: str
    variants: list[HookVariant]
    selected_text: str | None
    skip_adaptation: bool
    preferred_source: TextSource | None
    extraction_id: UUID | None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
