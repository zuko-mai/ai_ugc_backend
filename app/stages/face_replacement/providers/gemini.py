"""Gemini image-editing provider for face replacement."""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from app.core.logging import get_logger
from app.stages.face_replacement.exceptions import (
    GenerationFailedError,
    ProviderApiError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from app.stages.face_replacement.pricing import TokenUsage, estimate_cost
from app.stages.face_replacement.prompt import (
    CHARACTER_REPLACE_SYSTEM,
    build_character_replace_prompt,
    build_user_prompt,
)
from app.stages.face_replacement.providers.base import (
    FaceReplaceProviderRequest,
    FaceReplaceProviderResult,
    FaceReplacementProvider,
)

logger = get_logger(__name__)

# Re-export for older imports
__all__ = ["GeminiFaceReplacementProvider", "build_user_prompt", "build_character_replace_prompt"]


class GeminiFaceReplacementProvider(FaceReplacementProvider):
    name = "gemini"

    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ProviderNotConfiguredError(
                "GOOGLE_AI_API_KEY is not configured",
                details={"hint": "Set GOOGLE_AI_API_KEY in .env", "provider": self.name},
            )
        self.api_key = api_key

    async def replace_face(
        self, request: FaceReplaceProviderRequest
    ) -> FaceReplaceProviderResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._replace_sync, request),
                timeout=request.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            # Py3.10: asyncio.TimeoutError != builtin TimeoutError; 3.11+ they alias.
            raise ProviderTimeoutError(
                f"Gemini face replacement timed out after {request.timeout_seconds:.0f}s",
                details={
                    "provider": self.name,
                    "model": request.model_name,
                    "timeout_seconds": request.timeout_seconds,
                },
            ) from exc

    def _replace_sync(self, request: FaceReplaceProviderRequest) -> FaceReplaceProviderResult:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderNotConfiguredError(
                "google-genai package is not installed",
                details={"hint": "pip install google-genai"},
            ) from exc

        client = genai.Client(api_key=self.api_key)
        user_prompt = build_character_replace_prompt(
            remove_overlay_text=request.remove_overlay_text,
            extra_user_instructions=request.extra_user_instructions,
        )

        logger.info(
            "Gemini face replace model=%s base=%sx%s face=%sx%s",
            request.model_name,
            request.base_frame.width,
            request.base_frame.height,
            request.face_reference.width,
            request.face_reference.height,
        )

        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=request.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=user_prompt),
                            types.Part.from_text(
                                text=(
                                    "IMAGE 1 (FIRST) — selected_frame — PHOTO TO EDIT "
                                    "(master scene). Edit this image:"
                                )
                            ),
                            types.Part.from_bytes(
                                data=request.base_frame.data,
                                mime_type=request.base_frame.mime_type,
                            ),
                            types.Part.from_text(
                                text=(
                                    "IMAGE 2 (SECOND) — face_reference — IDENTITY REFERENCE ONLY "
                                    "(who the person should look like). Do not edit this image; "
                                    "do not copy its pose, clothing, crop or scale:"
                                )
                            ),
                            types.Part.from_bytes(
                                data=request.face_reference.data,
                                mime_type=request.face_reference.mime_type,
                            ),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=CHARACTER_REPLACE_SYSTEM,
                    temperature=0.2,
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderApiError(
                "Gemini face replacement API call failed",
                details={
                    "provider": self.name,
                    "model": request.model_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ) from exc
        latency_seconds = time.perf_counter() - started

        image_bytes, mime_type = _extract_image(response, model=request.model_name)
        width, height = _decode_dimensions(image_bytes)
        text_note = _extract_text_note(response)
        usage = _usage_from_response(model=request.model_name, response=response)

        warnings: list[str] = []
        if text_note:
            warnings.append(f"provider_note: {text_note[:300]}")

        logger.info(
            "Gemini face replace done model=%s tokens=%s cost_usd=%s latency_s=%.3f",
            request.model_name,
            usage.total_tokens,
            usage.estimated_cost_usd,
            latency_seconds,
        )
        return FaceReplaceProviderResult(
            image_bytes=image_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            model_name=request.model_name,
            provider_warnings=warnings,
            raw_metadata={"has_text_note": bool(text_note)},
            usage=usage,
            latency_seconds=round(latency_seconds, 4),
        )


def _extract_image(response, *, model: str) -> tuple[bytes, str]:  # noqa: ANN001
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if not data:
                continue
            if isinstance(data, str):
                import base64

                data = base64.b64decode(data)
            mime = getattr(inline, "mime_type", None) or "image/png"
            if not isinstance(data, (bytes, bytearray)):
                continue
            return bytes(data), mime

    raise GenerationFailedError(
        "Gemini returned no image in the response",
        details={"provider": "gemini", "model": model},
    )


def _extract_text_note(response) -> str | None:  # noqa: ANN001
    text = getattr(response, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    return None


def _usage_from_response(*, model: str, response) -> TokenUsage:  # noqa: ANN001
    meta = getattr(response, "usage_metadata", None)
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    candidates = int(getattr(meta, "candidates_token_count", 0) or 0) if meta else 0
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0) if meta else 0
    total = getattr(meta, "total_token_count", None) if meta else None
    total_int = int(total) if total is not None else None
    return estimate_cost(
        model=model,
        prompt_tokens=prompt,
        candidates_tokens=candidates,
        thoughts_tokens=thoughts,
        total_tokens=total_int,
    )


def _decode_dimensions(image_bytes: bytes) -> tuple[int, int]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        # Some providers return PNG that OpenCV can still fail on; try again via buffer.
        raise GenerationFailedError(
            "Generated image could not be decoded",
            details={"bytes": len(image_bytes)},
        )
    height, width = image.shape[:2]
    return int(width), int(height)
