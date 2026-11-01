"""Spoken-text ASR — Gemini audio primary; empty when no speech/audio."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.core.logging import get_logger
from app.stages.hook_text_engine.schemas import SpokenTextFragment

logger = get_logger(__name__)

_ASR_SYSTEM = """You are an ASR engine for short UGC video clips.
Transcribe ONLY spoken words that are clearly audible.
Do NOT invent dialogue. Do NOT copy on-screen captions unless they are also spoken.
If there is no speech (music-only, silence, noise), return {"segments":[]}.
Respond with JSON only:
{"segments":[{"text":"...","start_seconds":0.0,"end_seconds":1.2,"language":"en","confidence":0.0}]}
start_seconds/end_seconds are relative to the provided audio clip (clip starts at 0).
confidence is 0..1."""


async def extract_spoken_text(
    audio_path: Path | None,
    *,
    languages: list[str],
    api_key: str,
    model: str,
    clip_start_seconds: float,
) -> list[SpokenTextFragment]:
    if audio_path is None or not audio_path.is_file():
        return []
    if not api_key:
        logger.info("ASR skipped — GOOGLE_AI_API_KEY not configured")
        return []

    try:
        return await asyncio.to_thread(
            _gemini_asr,
            audio_path,
            languages,
            api_key,
            model,
            clip_start_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini ASR failed: %s", exc)
        return []


def _gemini_asr(
    audio_path: Path,
    languages: list[str],
    api_key: str,
    model: str,
    clip_start_seconds: float,
) -> list[SpokenTextFragment]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Preferred languages: {', '.join(languages)}. "
                            "Transcribe spoken speech in this audio clip."
                        )
                    ),
                    types.Part.from_bytes(
                        data=audio_path.read_bytes(),
                        mime_type=mime,
                    ),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=_ASR_SYSTEM,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json(response.text or "")
    segments = payload.get("segments") or []
    out: list[SpokenTextFragment] = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        rel_start = float(item.get("start_seconds", 0.0))
        rel_end = float(item.get("end_seconds", rel_start + 0.5))
        if rel_end < rel_start:
            rel_end = rel_start + 0.5
        lang = str(item.get("language") or (languages[0] if languages else "en"))
        conf = float(item.get("confidence", 0.7))
        out.append(
            SpokenTextFragment(
                text=text,
                start_seconds=round(clip_start_seconds + rel_start, 3),
                end_seconds=round(clip_start_seconds + rel_end, 3),
                language=lang,
                confidence=max(0.0, min(1.0, conf)),
            )
        )
    return out


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
            return {"segments": []}
        data = json.loads(match.group(0))
    if isinstance(data, list):
        return {"segments": data}
    return data if isinstance(data, dict) else {"segments": []}
