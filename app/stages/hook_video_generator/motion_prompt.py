"""Turn a motion plan into provider-agnostic image-to-video instructions.

Recognized plan keys (hook-analyzer shape, extra keys are ignored):

- emotion, emotion_intensity, gaze_direction, facial_expression
- motionPrompt | motion_prompt | prompt
- camera_motion
- motion_timeline[]: start_seconds/start, end_seconds/end,
  head_motion, eye_motion, mouth_motion, body_motion, emotion
"""

from __future__ import annotations

from typing import Any

from app.stages.hook_video_generator.instructions import MotionInstructions, TimelineCue

IDENTITY_LOCK = (
    "Preserve the person's appearance and visual identity for the entire clip. "
    "The face, hair, skin, clothing, accessories, age, and body proportions "
    "must match the first frame from start to finish."
)

CAMERA_LOCK = (
    "Keep the original composition and framing. "
    "Do not move the camera: no pan, tilt, zoom, dolly, orbit, or cut."
)

PERFORMANCE_RULE = (
    "Reproduce the described emotion, facial expression, eye movement, and head movement "
    "at the stated times. Do not invent extra gestures or expressions."
)

UNSPECIFIED_RULE = (
    'Signals marked "not clearly observed" or "undetermined" must stay unspecified. '
    "Do not invent them."
)

OPENING = (
    "Image-to-video from the provided first frame only. "
    "The first frame is the identity and composition reference for the whole clip."
)

NEGATIVE_PROMPT = (
    "different person, identity drift, face morph, changed age, extra people, "
    "extra limbs, distorted face, camera pan, camera tilt, zoom, dolly, orbit, "
    "cut, scene change, reframing, text overlay, watermark, logo, new objects"
)

NEGATIVE_PROMPT_WITH_CAMERA = (
    "different person, identity drift, face morph, changed age, extra people, "
    "extra limbs, distorted face, cut, scene change, text overlay, watermark, logo, "
    "camera movement other than the requested camera motion"
)

_LOCKED_CAMERA = {"none", "locked", "static", "lock", "fixed", "no camera movement"}


def compile_motion_instructions(
    motion_plan: dict[str, Any] | None,
    *,
    duration_seconds: float,
) -> MotionInstructions:
    """Compile a motion plan into timed identity-locked video instructions."""
    plan = motion_plan if isinstance(motion_plan, dict) else {}
    warnings: list[str] = []
    if motion_plan is not None and not isinstance(motion_plan, dict):
        warnings.append("motion_plan_ignored: expected an object")

    emotion = _clean(plan.get("emotion"))
    gaze = _clean(plan.get("gaze_direction"))
    expression = _clean(plan.get("facial_expression"))
    intensity, intensity_warning = _intensity(plan.get("emotion_intensity"))
    if intensity_warning:
        warnings.append(intensity_warning)
    notes = _first_text(plan, "motionPrompt", "motion_prompt", "prompt")
    camera_raw = _clean(plan.get("camera_motion"))
    timeline, timeline_warnings = _timeline(plan.get("motion_timeline"), duration_seconds)
    warnings.extend(timeline_warnings)

    camera_requested = bool(camera_raw) and camera_raw.lower() not in _LOCKED_CAMERA
    if camera_requested and camera_raw is not None:
        camera_lock = (
            f"Camera motion is limited to: {camera_raw}. "
            "Do not add any other camera movement, and do not change the framing beyond that."
        )
        warnings.append(f"camera_motion_requested: {camera_raw}")
        negative_prompt = NEGATIVE_PROMPT_WITH_CAMERA
    else:
        camera_lock = CAMERA_LOCK
        negative_prompt = NEGATIVE_PROMPT

    if not any((emotion, gaze, expression, notes, timeline, camera_raw)):
        warnings.append(
            "motion_plan_empty: no motion cues were provided; "
            "the clip is instructed to hold identity without invented movement"
        )

    prompt = _assemble(
        duration_seconds=duration_seconds,
        emotion=emotion,
        intensity=intensity,
        gaze=gaze,
        expression=expression,
        camera_lock=camera_lock,
        timeline=timeline,
        notes=notes,
    )
    return MotionInstructions(
        prompt=prompt,
        negative_prompt=negative_prompt,
        identity_lock=IDENTITY_LOCK,
        camera_lock=camera_lock,
        timeline=tuple(timeline),
        warnings=tuple(warnings),
    )


def _assemble(
    *,
    duration_seconds: float,
    emotion: str | None,
    intensity: float | None,
    gaze: str | None,
    expression: str | None,
    camera_lock: str,
    timeline: list[TimelineCue],
    notes: str | None,
) -> str:
    if emotion:
        emotion_line = (
            f"Overall emotion: {emotion} (intensity {intensity:.2f})."
            if intensity is not None
            else f"Overall emotion: {emotion}."
        )
    else:
        emotion_line = (
            "Overall emotion: unspecified. Keep the expression already visible in the first frame."
        )
    gaze_line = (
        f"Gaze: {gaze}."
        if gaze
        else "Gaze: keep the gaze already visible in the first frame."
    )
    expression_line = (
        f"Facial expression: {expression}."
        if expression
        else "Facial expression: keep the expression already visible in the first frame."
    )
    if timeline:
        cue_lines = "\n".join(f"- {cue.instruction}" for cue in timeline)
        timeline_block = f"Timeline:\n{cue_lines}"
    else:
        timeline_block = (
            "Timeline: no timed cues. Hold the first-frame pose and apply only the "
            "overall emotion and expression described above. "
            "Do not add head, eye, or body movement."
        )
    blocks = [
        OPENING,
        IDENTITY_LOCK,
        PERFORMANCE_RULE,
        camera_lock,
        UNSPECIFIED_RULE,
        f"The clip lasts exactly {duration_seconds:.2f} seconds.",
        emotion_line,
        gaze_line,
        expression_line,
        timeline_block,
    ]
    if notes:
        blocks.append(f"Additional motion notes:\n{notes}")
    return "\n\n".join(blocks)


def _timeline(raw: Any, duration_seconds: float) -> tuple[list[TimelineCue], list[str]]:
    warnings: list[str] = []
    if raw is None:
        return [], warnings
    if not isinstance(raw, list):
        return [], ["motion_timeline_ignored: expected a list"]

    cues: list[TimelineCue] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"motion_cue_skipped: item {index} is not an object")
            continue
        start = _number(_first(item, "start_seconds", "start", "from"))
        end = _number(_first(item, "end_seconds", "end", "to"))
        if start is None or end is None:
            warnings.append(f"motion_cue_skipped: item {index} is missing start or end")
            continue
        if start < 0:
            warnings.append(f"motion_cue_clamped: item {index} start {start:.2f}s raised to 0")
            start = 0.0
        if end <= start:
            warnings.append(f"motion_cue_skipped: item {index} ends at or before it starts")
            continue
        if start >= duration_seconds:
            warnings.append(
                f"motion_cue_outside: item {index} starts at {start:.2f}s, "
                f"after the {duration_seconds:.2f}s clip"
            )
            continue
        if end > duration_seconds:
            warnings.append(
                f"motion_cue_clamped: {start:.2f}-{end:.2f}s trimmed to {duration_seconds:.2f}s"
            )
            end = duration_seconds
        detail = _cue_detail(item)
        instruction = f"From {start:.2f}s to {end:.2f}s: {detail}."
        cues.append(TimelineCue(start_seconds=start, end_seconds=end, instruction=instruction))
    return cues, warnings


def _cue_detail(item: dict[str, Any]) -> str:
    parts = [
        ("head", _clean(item.get("head_motion")) or _clean(item.get("head"))),
        ("eyes", _clean(item.get("eye_motion")) or _clean(item.get("eye")) or _clean(item.get("gaze"))),
        ("mouth", _clean(item.get("mouth_motion")) or _clean(item.get("mouth"))),
        ("body", _clean(item.get("body_motion")) or _clean(item.get("body"))),
        ("emotion", _clean(item.get("emotion"))),
    ]
    rendered = [f"{label}: {value}" for label, value in parts if value]
    if not rendered:
        return "hold the current expression without adding movement"
    return "; ".join(rendered)


def _intensity(value: Any) -> tuple[float | None, str | None]:
    if value is None or value == "":
        return None, None
    number = _number(value)
    if number is None:
        return None, "emotion_intensity_ignored: value is not a number"
    if number < 0 or number > 1:
        return None, "emotion_intensity_ignored: value is outside 0..1"
    return number, None


def _first_text(plan: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        text = _clean(plan.get(key))
        if text:
            return text
    return None


def _first(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
