from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

AssetKind = Literal["face_reference", "product_demo", "music", "other"]


class Asset(BaseModel):
    asset_id: UUID
    url: str
    kind: AssetKind
    content_type: str | None = None
    filename: str
    size_bytes: int
    storage_key: str
    created_at: datetime


class AssetResponse(BaseModel):
    asset_id: UUID
    url: str
    kind: AssetKind
    content_type: str | None = None
    filename: str
    size_bytes: int


_DEFAULT_PRODUCT_DESCRIPTION = (
    "Saya is a pregnancy companion app for iOS and Android that helps expectant "
    "parents follow the journey week by week. It calculates gestational age and "
    "due date from common methods (LMP, conception date, ultrasound week, or "
    "estimated birth date) and shows weekly progress for baby, mom, nutrition, "
    "and tips. Users can track calendar events such as mood, medications, "
    "appointments, and notes, plus baby-care activities like sleep, diapers, "
    "breastfeeding, kicks, and contractions. The app includes expert articles, "
    "helper tools, baby-size and growth trackers, stories, and localized content "
    "in English, Russian, and Kazakh. Partners can link accounts to share "
    "pregnancy data and presence. An AI assistant answers pregnancy-related "
    "questions within a usage quota, and Premium unlocks extra features via "
    "App Store subscriptions. Content is informational only and is not a "
    "substitute for medical advice."
)


class Product(BaseModel):
    name: str | None = Field(default="Saya")
    description: str = Field(
        default=_DEFAULT_PRODUCT_DESCRIPTION,
        min_length=1,
    )
    target_audience: str | None = Field(default="expectant parents")
    language: str | None = Field(default="en")


class SourceVideo(BaseModel):
    video_id: UUID
    video_url: str


class AiCostRun(BaseModel):
    """One AI call for UI cost badges (reruns are separate entries)."""

    label: str
    model: str | None = None
    cost_usd: float | None = None
    is_rerun: bool = False
    created_at: datetime = Field(default_factory=_utc_now)


class HookAnalysis(BaseModel):
    analysis_key: str | None = None
    hook_start_seconds: float | None = None
    hook_end_seconds: float | None = None
    selected_frame_id: UUID | None = None
    selected_frame_url: str | None = None
    motion_prompt: str | None = None
    trimmed_video_url: str | None = None
    candidates: list[dict[str, Any]] | None = None
    ai_runs: list[AiCostRun] = Field(default_factory=list)


class FaceState(BaseModel):
    face_asset_id: UUID | None = None
    face_url: str | None = None
    generation_id: UUID | None = None
    generated_frame_url: str | None = None
    user_prompt: str | None = None
    ai_runs: list[AiCostRun] = Field(default_factory=list)


class HookVideoState(BaseModel):
    job_id: UUID | None = None
    video_url: str | None = None
    asset_id: UUID | None = None
    motion_prompt: str | None = None
    ai_runs: list[AiCostRun] = Field(default_factory=list)


class HookTextState(BaseModel):
    extraction_id: UUID | None = None
    adaptation_id: UUID | None = None
    selected_text: str | None = None
    variants: list[dict[str, Any]] | None = None
    ai_runs: list[AiCostRun] = Field(default_factory=list)


class RenderState(BaseModel):
    render_id: UUID | None = None
    asset_id: UUID | None = None
    video_url: str | None = None


class AsyncVideoState(BaseModel):
    job_id: UUID | None = None
    asset_id: UUID | None = None
    video_url: str | None = None


class CompositionState(AsyncVideoState):
    demo_asset_id: UUID | None = None


class MusicState(AsyncVideoState):
    music_asset_id: UUID | None = None


class PipelineSession(BaseModel):
    session_id: UUID
    created_at: datetime
    updated_at: datetime
    step: int = Field(0, ge=0, le=8)
    product: Product | None = None
    source_video: SourceVideo | None = None
    hook_analysis: HookAnalysis | None = None
    face: FaceState | None = None
    hook_video: HookVideoState | None = None
    hook_text: HookTextState | None = None
    render: RenderState | None = None
    composition: CompositionState | None = None
    music: MusicState | None = None


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=1)


class CandidateOptions(BaseModel):
    hook_start_seconds: float = Field(0, ge=0)
    hook_end_seconds: float = Field(4, gt=0)
    candidate_count: int = Field(5, ge=1, le=10)


class SelectFrameRequest(BaseModel):
    selected_frame_id: UUID
    generate_motion_plan: bool = True
    start_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Optional. Defaults to the candidates hook interval start.",
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        le=15,
        description="Optional. Defaults to the candidates hook interval length.",
    )


class HookVideoOptions(BaseModel):
    prompt: str | None = None
    duration_seconds: float = Field(3, ge=2, le=4)
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    output_resolution: str = "1080x1920"
    fps: int = Field(30, ge=12, le=60)
    model_provider: str | None = None
    model_name: str | None = None


class HookTextOptions(BaseModel):
    start_seconds: float = Field(0, ge=0)
    end_seconds: float = Field(4, gt=0)
    preferred_source: Literal["on_screen", "spoken", "composed"] | None = None
    hook_goal: Literal[
        "curiosity", "urgency", "pain", "benefit", "social_proof", "question", "contrast"
    ] = "curiosity"
    variants_count: int = Field(5, ge=1, le=10)


class SelectTextRequest(BaseModel):
    variant_id: UUID | None = None
    manual_text: str | None = None
    use_original: bool = False


class RenderOptions(BaseModel):
    output_format: Literal["mp4", "mov"] = "mp4"
    skip_text: bool = False


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict | None = None
