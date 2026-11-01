"""Deterministic offline provider for tests and local dry-runs.

Returns the base frame unchanged so the pipeline (persist, validate, text check)
can be exercised without a live model API. Not suitable for production output.
"""

from __future__ import annotations

import time

from app.stages.face_replacement.pricing import empty_usage
from app.stages.face_replacement.providers.base import (
    FaceReplaceProviderRequest,
    FaceReplaceProviderResult,
    FaceReplacementProvider,
)


class StubFaceReplacementProvider(FaceReplacementProvider):
    name = "stub"

    async def replace_face(
        self, request: FaceReplaceProviderRequest
    ) -> FaceReplaceProviderResult:
        started = time.perf_counter()
        model_name = request.model_name or "stub-passthrough"
        latency = time.perf_counter() - started
        return FaceReplaceProviderResult(
            image_bytes=request.base_frame.data,
            mime_type=request.base_frame.mime_type,
            width=request.base_frame.width,
            height=request.base_frame.height,
            model_name=model_name,
            provider_warnings=[
                "stub_provider: base frame returned unchanged; no real face swap performed",
            ],
            raw_metadata={"mode": "passthrough"},
            usage=empty_usage(model=model_name),
            latency_seconds=round(latency, 4),
        )
