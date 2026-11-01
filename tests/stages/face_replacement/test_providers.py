import asyncio

import pytest

from app.core.config import Settings
from app.stages.face_replacement.exceptions import (
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from app.stages.face_replacement.providers.base import (
    FaceReplaceProviderRequest,
    FaceReplaceProviderResult,
    FaceReplacementProvider,
    ProviderImage,
)
from app.stages.face_replacement.prompt import build_character_replace_prompt, build_user_prompt
from app.stages.face_replacement.providers.registry import get_provider, register_provider
from app.stages.face_replacement.providers.stub import StubFaceReplacementProvider


def _img(data: bytes = b"fake", w: int = 100, h: int = 100) -> ProviderImage:
    return ProviderImage(
        data=data,
        mime_type="image/jpeg",
        width=w,
        height=h,
        source_url="http://example/x.jpg",
    )


def test_build_user_prompt_includes_text_removal() -> None:
    prompt = build_character_replace_prompt(remove_overlay_text=True)
    assert "IMAGE 1 = FIRST image = selected_frame = the photo you MUST edit" in prompt
    assert "IMAGE 2 = SECOND image = face_reference = identity reference ONLY" in prompt
    assert "Remove all added text" in prompt
    assert "Do not recreate or replace the removed text" in prompt
    assert "photo to edit" in prompt
    assert "identity reference" in prompt
    assert "HEAD AND BODY PROPORTIONS" in prompt
    assert "Keep the EXACT clothing from the FIRST image" in prompt
    assert "SKIN TONE CONSISTENCY" in prompt
    assert "hands and arms must not look pale" in prompt
    assert "No obvious face-swap effect" in prompt
    assert "Output a single edited version of IMAGE 1" in prompt


def test_build_user_prompt_keeps_text_when_not_requested() -> None:
    prompt = build_user_prompt(
        remove_overlay_text=False,
        preserve_original_composition=False,
        preserve_pose=False,
        preserve_lighting=False,
    )
    assert "Leave any existing overlay text" in prompt
    assert "Remove all added text" not in prompt


@pytest.mark.asyncio
async def test_stub_provider_passthrough() -> None:
    provider = StubFaceReplacementProvider()
    base = _img(b"frame-bytes", 640, 480)
    result = await provider.replace_face(
        FaceReplaceProviderRequest(
            base_frame=base,
            face_reference=_img(b"face-bytes", 256, 256),
            remove_overlay_text=True,
            preserve_original_composition=True,
            preserve_pose=True,
            preserve_lighting=True,
            model_name="stub-passthrough",
            timeout_seconds=5,
        )
    )
    assert result.image_bytes == b"frame-bytes"
    assert result.width == 640
    assert result.height == 480
    assert result.provider_warnings


def test_unknown_provider_raises(settings: Settings) -> None:
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("nope", settings)


def test_gemini_requires_api_key(settings: Settings) -> None:
    settings_no_key = settings.model_copy(update={"google_ai_api_key": ""})
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("gemini", settings_no_key)


@pytest.mark.asyncio
async def test_provider_timeout_mapped(settings: Settings) -> None:
    class SlowProvider(FaceReplacementProvider):
        name = "slow"

        async def replace_face(
            self, request: FaceReplaceProviderRequest
        ) -> FaceReplaceProviderResult:
            await asyncio.sleep(10)
            return FaceReplaceProviderResult(
                image_bytes=b"x",
                mime_type="image/png",
                width=1,
                height=1,
                model_name="slow",
            )

    register_provider("slow", lambda _s: SlowProvider())

    from app.stages.face_replacement.providers.gemini import GeminiFaceReplacementProvider

    class Timed(GeminiFaceReplacementProvider):
        def __init__(self) -> None:
            self.api_key = "test"

        def _replace_sync(self, request: FaceReplaceProviderRequest) -> FaceReplaceProviderResult:
            import time

            time.sleep(2)
            return FaceReplaceProviderResult(
                image_bytes=b"x",
                mime_type="image/png",
                width=1,
                height=1,
                model_name="x",
            )

    provider = Timed()
    with pytest.raises(ProviderTimeoutError):
        await provider.replace_face(
            FaceReplaceProviderRequest(
                base_frame=_img(),
                face_reference=_img(),
                remove_overlay_text=True,
                preserve_original_composition=True,
                preserve_pose=True,
                preserve_lighting=True,
                model_name="x",
                timeout_seconds=0.05,
            )
        )
