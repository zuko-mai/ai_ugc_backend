from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.stages.face_replacement.pricing import TokenUsage


@dataclass(frozen=True)
class ProviderImage:
    """Decoded image payload passed to a model provider."""

    data: bytes
    mime_type: str
    width: int
    height: int
    source_url: str


@dataclass(frozen=True)
class FaceReplaceProviderRequest:
    base_frame: ProviderImage
    face_reference: ProviderImage
    remove_overlay_text: bool
    preserve_original_composition: bool
    preserve_pose: bool
    preserve_lighting: bool
    model_name: str
    timeout_seconds: float
    extra_user_instructions: str | None = None


@dataclass(frozen=True)
class FaceReplaceProviderResult:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    model_name: str
    provider_warnings: list[str] = field(default_factory=list)
    raw_metadata: dict | None = None
    usage: TokenUsage | None = None
    latency_seconds: float | None = None


class FaceReplacementProvider(ABC):
    """Adapter boundary between business logic and a concrete model API."""

    name: str

    @abstractmethod
    async def replace_face(
        self, request: FaceReplaceProviderRequest
    ) -> FaceReplaceProviderResult:
        """Swap face identity onto the base frame while preserving scene constraints."""
