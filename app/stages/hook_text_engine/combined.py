"""Single Gemini call: watch hook clip → OCR + ASR + product-adapted variants."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.stages.hook_text_engine.adaptation import _finalize_variant
from app.stages.hook_text_engine.exceptions import AdaptationError, ExtractionError
from app.stages.hook_text_engine.schemas import (
    HookGoal,
    HookVariant,
    OnScreenTextFragment,
    SpokenTextFragment,
    TextSource,
)

logger = get_logger(__name__)

_COMBINED_SYSTEM = """You analyze a short UGC hook video clip for marketing reuse.

Do THREE jobs in one response, keeping fields SEPARATE — never mix them:

1) on_screen_text: ONLY burned-in / overlay captions visible in the video. Empty list if none.
2) spoken_text: ONLY clearly audible spoken words. Empty list if none / music-only.
3) variants: rewrite the hook for the given product. Preserve the original psychological mechanic
   and similar length/tone. NEVER invent features, %, reviews, or proof not in product_description.

Choose original_text as the best single hook string to adapt from:
- prefer on-screen caption if it is a clear hook
- else spoken
- never silently merge on-screen and spoken into one sentence

Output JSON only with this exact shape:
{
  "on_screen_text":[{"text":"...","start_seconds":0.0,"end_seconds":0.5,"confidence":0.0,"language":"en"}],
  "spoken_text":[{"text":"...","start_seconds":0.0,"end_seconds":1.0,"confidence":0.0,"language":"en"}],
  "preferred_source":"on_screen",
  "original_text":"...",
  "variants":[{"hook_text":"...","hook_type":"curiosity","structure":"question"}]
}
preferred_source must be one of: on_screen, spoken.
hook_type must be one of: curiosity, urgency, pain, benefit, social_proof, question, contrast.
"""


@dataclass(frozen=True)
class CombinedExtractAdaptResult:
    on_screen_text: list[OnScreenTextFragment]
    spoken_text: list[SpokenTextFragment]
    preferred_source: TextSource
    original_text: str
    variants: list[HookVariant]
    model: str
    estimated_cost_usd: float | None = None


async def extract_and_adapt_combined(
    *,
    video_path: Path,
    product_name: str,
    product_description: str,
    target_audience: str,
    language: str,
    hook_goal: HookGoal,
    variants_count: int,
    languages: list[str],
    start_seconds: float,
    end_seconds: float,
    api_key: str,
    model: str,
    fallback_model: str | None = None,
) -> CombinedExtractAdaptResult:
    if not api_key:
        raise ExtractionError(
            "GOOGLE_AI_API_KEY is not configured for combined hook-text",
            details={"hint": "Set GOOGLE_AI_API_KEY in .env"},
        )
    if not video_path.is_file():
        raise ExtractionError(
            "Hook clip missing for combined extract+adapt",
            details={"path": str(video_path)},
        )

    models = [model]
    if fallback_model and fallback_model != model:
        models.append(fallback_model)

    last_error: Exception | None = None
    for candidate in models:
        try:
            return await asyncio.to_thread(
                _run_sync,
                video_path,
                product_name,
                product_description,
                target_audience,
                language,
                hook_goal,
                variants_count,
                languages,
                start_seconds,
                end_seconds,
                api_key,
                candidate,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Combined hook-text failed model=%s error=%s",
                candidate,
                exc,
            )
    raise AdaptationError(
        "Combined hook-text extract+adapt failed",
        details={"error": str(last_error), "models": models},
    ) from last_error


def _run_sync(
    video_path: Path,
    product_name: str,
    product_description: str,
    target_audience: str,
    language: str,
    hook_goal: HookGoal,
    variants_count: int,
    languages: list[str],
    start_seconds: float,
    end_seconds: float,
    api_key: str,
    model: str,
) -> CombinedExtractAdaptResult:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(video_path))
        uploaded = _wait_until_active(client, uploaded)
        duration = max(0.1, end_seconds - start_seconds)
        user = (
            f"Clip window: {start_seconds:.2f}s → {end_seconds:.2f}s "
            f"(duration {duration:.2f}s).\n"
            f"Preferred languages: {', '.join(languages)}.\n"
            f"Output language for variants: {language}.\n"
            f"Product name: {product_name}\n"
            f"Product description: {product_description}\n"
            f"Target audience: {target_audience}\n"
            f"Preferred hook_goal: {hook_goal}\n"
            f"Generate exactly {variants_count} variants."
        )
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=uploaded.mime_type or "video/mp4",
                        ),
                        types.Part.from_text(text=user),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=_COMBINED_SYSTEM,
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        payload = _parse_json(response.text or "")
        return _materialize(
            payload,
            product_name=product_name,
            product_description=product_description,
            target_audience=target_audience,
            model=model,
            clip_start=start_seconds,
            estimated_cost_usd=_estimate_cost(model, response),
        )
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to delete uploaded hook-text clip", exc_info=True)


def _materialize(
    payload: dict,
    *,
    product_name: str,
    product_description: str,
    target_audience: str,
    model: str,
    clip_start: float,
    estimated_cost_usd: float | None,
) -> CombinedExtractAdaptResult:
    on_screen_raw = payload.get("on_screen_text") or []
    spoken_raw = payload.get("spoken_text") or []
    if not isinstance(on_screen_raw, list):
        on_screen_raw = []
    if not isinstance(spoken_raw, list):
        spoken_raw = []

    on_screen: list[OnScreenTextFragment] = []
    for item in on_screen_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start_seconds") or clip_start)
        end = float(item.get("end_seconds") or start)
        on_screen.append(
            OnScreenTextFragment(
                text=text,
                start_seconds=max(0.0, start),
                end_seconds=max(start, end),
                bbox=[0, 0, 0, 0],
                confidence=float(item.get("confidence") or 0.5),
                language=item.get("language"),
            )
        )

    spoken: list[SpokenTextFragment] = []
    for item in spoken_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start_seconds") or clip_start)
        end = float(item.get("end_seconds") or start)
        spoken.append(
            SpokenTextFragment(
                text=text,
                start_seconds=max(0.0, start),
                end_seconds=max(start, end),
                language=str(item.get("language") or "en"),
                confidence=float(item.get("confidence") or 0.5),
            )
        )

    preferred = payload.get("preferred_source")
    if preferred not in {"on_screen", "spoken"}:
        preferred = "on_screen" if on_screen else "spoken" if spoken else "composed"

    original = str(payload.get("original_text") or "").strip()
    if not original:
        if preferred == "spoken" and spoken:
            original = " ".join(s.text for s in spoken)
        elif on_screen:
            original = " ".join(o.text for o in on_screen)
        elif spoken:
            original = " ".join(s.text for s in spoken)
    if not original:
        raise ExtractionError(
            "Combined hook-text found no on-screen or spoken hook text",
            details={"model": model},
        )

    variants_raw = payload.get("variants") or []
    if not isinstance(variants_raw, list) or not variants_raw:
        raise AdaptationError(
            "Combined hook-text returned no variants",
            details={"model": model},
        )

    variants: list[HookVariant] = []
    for item in variants_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("hook_text") or "").strip()
        if not text:
            continue
        variants.append(
            _finalize_variant(
                text=text,
                hook_type=str(item.get("hook_type") or "curiosity"),
                structure=str(item.get("structure") or ""),
                product_name=product_name,
                product_description=product_description,
                target_audience=target_audience,
                variant_id=uuid.uuid4(),
            )
        )
    if not variants:
        raise AdaptationError(
            "Combined hook-text variants were empty after validation",
            details={"model": model},
        )

    source: TextSource = preferred  # type: ignore[assignment]
    return CombinedExtractAdaptResult(
        on_screen_text=on_screen,
        spoken_text=spoken,
        preferred_source=source,
        original_text=original,
        variants=variants,
        model=model,
        estimated_cost_usd=estimated_cost_usd,
    )


def _wait_until_active(client, uploaded, *, timeout_seconds: float = 120.0):  # noqa: ANN001
    import time

    started = time.perf_counter()
    current = uploaded
    while getattr(getattr(current, "state", None), "name", None) == "PROCESSING":
        if time.perf_counter() - started > timeout_seconds:
            raise ExtractionError("Timed out waiting for Gemini file processing")
        time.sleep(1.0)
        current = client.files.get(name=current.name)
    state = getattr(getattr(current, "state", None), "name", None)
    if state and state != "ACTIVE":
        raise ExtractionError(
            "Gemini rejected uploaded hook clip",
            details={"state": state},
        )
    return current


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
            return {}
        data = json.loads(match.group(0))
    return data if isinstance(data, dict) else {}


def _estimate_cost(model: str, response) -> float | None:  # noqa: ANN001
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    candidates = int(getattr(meta, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    # Approx paid rates; flash-lite class.
    rates = {
        "gemini-3.1-flash-lite": (0.25, 1.50),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-3.8-flash": (0.75, 3.75),
    }
    inp, out = rates.get(model, (0.25, 1.50))
    return round((prompt / 1_000_000) * inp + ((candidates + thoughts) / 1_000_000) * out, 8)
