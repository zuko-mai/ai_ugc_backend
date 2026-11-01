import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import GeminiMotionPromptError
from app.stages.hook_analyzer.gemini_pricing import TokenUsage, estimate_cost

logger = get_logger(__name__)

KLING_MOTION_SYSTEM_PROMPT = """You are a motion-direction specialist for Kling 3.0 (AI video).

Watch the provided UGC clip carefully. Your job is to write ONE motionPrompt that Kling can use later to recreate the SAME human actions, micro-movements, and emotional performance as naturally as possible.

Rules:
1. Describe ONLY what is visibly/audibly present. Do not invent props, locations, dialogue, or actions that are not in the clip.
2. Focus on: facial expression, emotion and its intensity over time, gaze, head/neck, shoulders, torso, hands/arms if visible, breathing/micro-movements, speaking vs silent mouth motion, camera framing feel (selfie / talking-head / etc.).
3. Write chronologically for the clip duration (start → middle → end). Mention timing softly ("at the start…", "then…", "toward the end…") without fake millisecond precision.
4. Style: natural cinematic English, one cohesive paragraph (or 2 short paragraphs max). No bullet lists, no JSON, no markdown, no camera technical jargon Kling does not need.
5. Optimize for human realism: subtle asymmetries, soft transitions, avoid robotic or exaggerated animation language.
6. Do not mention Kling, prompts, AI, or that you are analyzing a video. Output only the motionPrompt text itself.
"""


@dataclass(frozen=True)
class GeminiMotionResult:
    motion_prompt: str
    usage: TokenUsage


async def generate_kling_motion_prompt(
    *,
    video_path: Path,
    api_key: str,
    model: str,
    start_seconds: float,
    duration_seconds: float,
) -> GeminiMotionResult:
    """Upload a short trimmed clip to Gemini and return motionPrompt + usage/cost."""
    if not api_key:
        raise GeminiMotionPromptError(
            "GOOGLE_AI_API_KEY is not configured",
            details={"hint": "Set GOOGLE_AI_API_KEY in .env"},
        )
    if not video_path.is_file():
        raise GeminiMotionPromptError(
            "Trimmed video file missing for Gemini analysis",
            details={"path": str(video_path)},
        )

    return await asyncio.to_thread(
        _generate_sync,
        video_path,
        api_key,
        model,
        start_seconds,
        duration_seconds,
    )


def _generate_sync(
    video_path: Path,
    api_key: str,
    model: str,
    start_seconds: float,
    duration_seconds: float,
) -> GeminiMotionResult:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiMotionPromptError(
            "google-genai package is not installed",
            details={"hint": "pip install google-genai"},
        ) from exc

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        logger.info(
            "Uploading trimmed clip to Gemini model=%s path=%s size=%s",
            model,
            video_path.name,
            video_path.stat().st_size,
        )
        uploaded = client.files.upload(file=str(video_path))
        uploaded = _wait_until_active(client, uploaded)

        user_prompt = (
            f"This clip is a {duration_seconds:.2f}s UGC segment starting at "
            f"{start_seconds:.2f}s of the original video. "
            "Write the Kling 3.0 motionPrompt now."
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
                        types.Part.from_text(text=user_prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=KLING_MOTION_SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise GeminiMotionPromptError(
                "Gemini returned an empty motionPrompt",
                details={"model": model},
            )

        usage = _usage_from_response(model=model, response=response)
        logger.info(
            "Gemini motionPrompt ready model=%s chars=%s tokens=%s cost_usd=%s",
            model,
            len(text),
            usage.total_tokens,
            usage.estimated_cost_usd,
        )
        return GeminiMotionResult(motion_prompt=text, usage=usage)
    except GeminiMotionPromptError:
        raise
    except Exception as exc:
        raise GeminiMotionPromptError(
            "Gemini motionPrompt generation failed",
            details={"model": model, "error": str(exc)},
        ) from exc
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                client.files.delete(name=uploaded.name)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("Failed to delete Gemini temp file: %s", cleanup_exc)


def _usage_from_response(*, model: str, response) -> TokenUsage:
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


def _wait_until_active(client, uploaded, *, timeout_seconds: float = 120.0):
    deadline = time.monotonic() + timeout_seconds
    state_name = _state_name(uploaded)
    while state_name in {"PROCESSING", "STATE_UNSPECIFIED", ""}:
        if time.monotonic() > deadline:
            raise GeminiMotionPromptError(
                "Timed out waiting for Gemini file processing",
                details={"state": state_name},
            )
        time.sleep(1.5)
        uploaded = client.files.get(name=uploaded.name)
        state_name = _state_name(uploaded)

    if state_name not in {"ACTIVE", "ACTIVE_STATE", "SUCCEEDED"}:
        if state_name != "ACTIVE":
            raise GeminiMotionPromptError(
                "Gemini file processing failed",
                details={"state": state_name},
            )
    return uploaded


def _state_name(uploaded) -> str:
    state = getattr(uploaded, "state", None)
    if state is None:
        return ""
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)
