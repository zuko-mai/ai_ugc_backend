"""On-screen text OCR — Gemini vision primary, OpenCV local detector fallback."""

from __future__ import annotations

import asyncio
import json
import re

import cv2
import numpy as np

from app.core.logging import get_logger
from app.stages.hook_text_engine.frames import ExtractedFrame
from app.stages.hook_text_engine.schemas import OnScreenTextFragment

logger = get_logger(__name__)

_OCR_SYSTEM = """You are an OCR engine for UGC video frames.
Extract ONLY text that is visibly overlaid or burned into the image (captions, titles, stickers).
Do NOT invent text. Do NOT transcribe spoken words. If no on-screen text is visible, return [].
Respond with JSON only:
{"fragments":[{"text":"...","bbox":[x,y,w,h],"confidence":0.0,"language":"en"}]}
bbox is pixel coordinates relative to the image: x,y,width,height.
confidence is 0..1."""


async def extract_on_screen_text(
    frames: list[ExtractedFrame],
    *,
    languages: list[str],
    api_key: str,
    model: str,
    frame_window_seconds: float = 0.5,
) -> list[OnScreenTextFragment]:
    """Run OCR across sampled frames; merge near-duplicates by text."""
    if not frames:
        return []

    if api_key:
        try:
            fragments = await asyncio.to_thread(
                _gemini_ocr,
                frames,
                languages,
                api_key,
                model,
                frame_window_seconds,
            )
            if fragments:
                return _dedupe_fragments(fragments)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini OCR failed, using local fallback: %s", exc)

    local = _local_ocr(frames, languages=languages, frame_window_seconds=frame_window_seconds)
    return _dedupe_fragments(local)


def _gemini_ocr(
    frames: list[ExtractedFrame],
    languages: list[str],
    api_key: str,
    model: str,
    frame_window_seconds: float,
) -> list[OnScreenTextFragment]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # Cap frames sent to the model for cost/latency.
    sample = frames if len(frames) <= 6 else [frames[i] for i in _even_indices(len(frames), 6)]

    parts: list = [
        types.Part.from_text(
            text=(
                f"Preferred languages: {', '.join(languages)}. "
                "Extract on-screen overlay text from each frame below. "
                "Return one JSON object covering all frames; include start timing hints "
                "via frame order only — timestamps are assigned server-side."
            )
        )
    ]
    for frame in sample:
        if frame.path is not None and frame.path.is_file():
            image_bytes = frame.path.read_bytes()
        else:
            ok, buf = cv2.imencode(".jpg", frame.image_bgr)
            if not ok:
                continue
            image_bytes = buf.tobytes()
        parts.append(
            types.Part.from_text(
                text=f"frame_timestamp_seconds={frame.timestamp_seconds:.3f}"
            )
        )
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=_OCR_SYSTEM,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json(response.text or "")
    raw_frags = payload.get("fragments") or []
    out: list[OnScreenTextFragment] = []
    for idx, item in enumerate(raw_frags):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        ts = float(
            item.get(
                "start_seconds",
                sample[min(idx, len(sample) - 1)].timestamp_seconds,
            )
        )
        bbox = _normalize_bbox(item.get("bbox"), sample[0].image_bgr.shape)
        conf = float(item.get("confidence", 0.7))
        lang = item.get("language") or (languages[0] if languages else None)
        out.append(
            OnScreenTextFragment(
                text=text,
                start_seconds=round(ts, 3),
                end_seconds=round(ts + frame_window_seconds, 3),
                bbox=bbox,
                confidence=max(0.0, min(1.0, conf)),
                language=lang,
            )
        )
    return out


def _local_ocr(
    frames: list[ExtractedFrame],
    *,
    languages: list[str],
    frame_window_seconds: float,
) -> list[OnScreenTextFragment]:
    """Detect text-like regions; read with pytesseract when available."""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        pytesseract = None

    fragments: list[OnScreenTextFragment] = []
    for frame in frames:
        h, w = frame.image_bgr.shape[:2]
        regions = _detect_text_regions(frame.image_bgr)
        for x, y, rw, rh in regions:
            crop = frame.image_bgr[y : y + rh, x : x + rw]
            text = ""
            conf = 0.35
            if pytesseract is not None and crop.size > 0:
                try:
                    data = pytesseract.image_to_data(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        output_type=pytesseract.Output.DICT,
                        lang="+".join(languages) if languages else "eng",
                    )
                    words = [
                        data["text"][i]
                        for i in range(len(data["text"]))
                        if str(data["text"][i]).strip()
                        and int(data.get("conf", [0])[i] or 0) > 40
                    ]
                    if words:
                        text = " ".join(words).strip()
                        confs = [
                            int(data["conf"][i])
                            for i in range(len(data["text"]))
                            if str(data["text"][i]).strip()
                        ]
                        conf = max(0.0, min(1.0, (sum(confs) / max(1, len(confs))) / 100.0))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("pytesseract failed: %s", exc)

            if not text:
                # No readable text without tesseract — skip empty detections.
                continue

            fragments.append(
                OnScreenTextFragment(
                    text=text,
                    start_seconds=round(frame.timestamp_seconds, 3),
                    end_seconds=round(frame.timestamp_seconds + frame_window_seconds, 3),
                    bbox=[int(x), int(y), int(rw), int(rh)],
                    confidence=conf,
                    language=languages[0] if languages else None,
                )
            )
            _ = (w, h)  # keep dims referenced for linters
    return fragments


def _detect_text_regions(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = rw * rh
        if area < (w * h) * 0.002 or area > (w * h) * 0.45:
            continue
        aspect = rw / max(rh, 1)
        if aspect < 1.2 or aspect > 18:
            continue
        regions.append((x, y, rw, rh))
    regions.sort(key=lambda r: (r[1], r[0]))
    return regions[:12]


def _dedupe_fragments(fragments: list[OnScreenTextFragment]) -> list[OnScreenTextFragment]:
    seen: dict[str, OnScreenTextFragment] = {}
    for frag in fragments:
        key = re.sub(r"\s+", " ", frag.text.strip().lower())
        if not key:
            continue
        existing = seen.get(key)
        if existing is None or frag.confidence > existing.confidence:
            seen[key] = frag
    return list(seen.values())


def _normalize_bbox(raw, shape) -> list[int]:  # noqa: ANN001
    h, w = shape[:2]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return [0, 0, int(w), int(min(h, 50))]
    x, y, bw, bh = (int(float(v)) for v in raw)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return [x, y, bw, bh]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"fragments": []}
        data = json.loads(match.group(0))
    if isinstance(data, list):
        return {"fragments": data}
    return data if isinstance(data, dict) else {"fragments": []}


def _even_indices(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]
