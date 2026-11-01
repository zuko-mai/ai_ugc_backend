"""Post-generation residual overlay-text detection.

Models do not reliably remove burned-in captions. This module runs a best-effort
check and returns warnings — it does not hard-fail the generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextCheckResult:
    residual_text_likely: bool
    confidence: float
    method: str
    details: str | None = None


async def check_residual_overlay_text(
    image_bytes: bytes,
    *,
    api_key: str = "",
    model: str = "",
) -> TextCheckResult:
    """Return whether residual on-screen text is likely still present."""
    local = await asyncio.to_thread(_local_text_heuristic, image_bytes)
    if not api_key:
        return local

    try:
        gemini = await asyncio.to_thread(
            _gemini_text_check, image_bytes, api_key, model or "gemini-3.8-flash"
        )
        # Prefer Gemini when it is confident; otherwise keep local signal.
        if gemini.confidence >= 0.55:
            return gemini
        if local.residual_text_likely and local.confidence >= 0.45:
            return TextCheckResult(
                residual_text_likely=True,
                confidence=max(local.confidence, gemini.confidence),
                method="local+gemini",
                details=f"local={local.details}; gemini={gemini.details}",
            )
        return gemini
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini residual-text check failed: %s", exc)
        return local


def _local_text_heuristic(image_bytes: bytes) -> TextCheckResult:
    """Detect caption-like high-contrast horizontal bands via MSER/edges."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return TextCheckResult(
            residual_text_likely=False,
            confidence=0.0,
            method="local",
            details="decode_failed",
        )

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Caption bands commonly sit in the lower / upper thirds.
    band_h = max(24, height // 5)
    bands = [
        ("top", gray[0:band_h, :]),
        ("bottom", gray[height - band_h : height, :]),
        ("mid", gray[height // 3 : 2 * height // 3, :]),
    ]

    score = 0.0
    hits: list[str] = []
    for name, band in bands:
        edges = cv2.Canny(band, 80, 180)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        # Horizontal projection peaks suggest line-like text.
        projection = edges.mean(axis=1)
        peak = float(projection.max()) if projection.size else 0.0
        mean = float(projection.mean()) if projection.size else 0.0
        peak_ratio = peak / (mean + 1e-6)

        if edge_ratio > 0.08 and peak_ratio > 2.2:
            score = max(score, min(1.0, edge_ratio * 4.0 + (peak_ratio - 2.0) * 0.15))
            hits.append(f"{name}:edge={edge_ratio:.3f},peak={peak_ratio:.2f}")

        # Bright/dark solid caption bars
        band_std = float(band.std())
        band_mean = float(band.mean())
        if band_std < 28 and (band_mean < 40 or band_mean > 210) and edge_ratio > 0.03:
            score = max(score, 0.55)
            hits.append(f"{name}:solid_bar mean={band_mean:.0f}")

    # Also try MSER regions with text-like aspect ratios across the full frame.
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        textish = 0
        for pts in regions[:400]:
            x, y, w, h = cv2.boundingRect(pts)
            if h < 8 or w < 8:
                continue
            aspect = w / float(h)
            if 0.15 <= aspect <= 8.0 and h < height * 0.2 and w < width * 0.7:
                textish += 1
        if textish >= 25:
            score = max(score, min(0.85, textish / 80.0))
            hits.append(f"mser_regions={textish}")
    except Exception:  # noqa: BLE001
        pass

    likely = score >= 0.45
    return TextCheckResult(
        residual_text_likely=likely,
        confidence=round(score, 3),
        method="local",
        details="; ".join(hits) if hits else "no_text_like_bands",
    )


def _gemini_text_check(image_bytes: bytes, api_key: str, model: str) -> TextCheckResult:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = (
        "Does this image still contain burned-in / overlaid caption text, titles, "
        "stickers with words, or watermarks? Answer ONLY JSON: "
        '{"has_overlay_text": true|false, "confidence": 0.0-1.0, "note": "short"}'
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
            )
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = (response.text or "").strip()
    has_text, confidence, note = _parse_json_flag(text)
    return TextCheckResult(
        residual_text_likely=has_text,
        confidence=confidence,
        method="gemini",
        details=note or text[:200],
    )


def _parse_json_flag(text: str) -> tuple[bool, float, str | None]:
    import json
    import re

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        lowered = text.lower()
        if "true" in lowered:
            return True, 0.5, text[:120]
        if "false" in lowered:
            return False, 0.5, text[:120]
        return False, 0.0, text[:120]
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, 0.0, text[:120]
    has = bool(payload.get("has_overlay_text", False))
    conf = float(payload.get("confidence", 0.5))
    note = payload.get("note")
    return has, max(0.0, min(1.0, conf)), str(note) if note else None
