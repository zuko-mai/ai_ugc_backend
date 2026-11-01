from uuid import uuid4

from app.stages.hook_analyzer.motion import build_motion_plan
from app.stages.hook_analyzer.schemas import FaceObservation, HookInterval


def test_motion_plan_is_ai_prompt_focused() -> None:
    video_id = uuid4()
    observations = [
        FaceObservation(timestamp_seconds=0.0, face_detected=False, sharpness=10.0),
        FaceObservation(timestamp_seconds=0.5, face_detected=False, sharpness=12.0),
    ]
    plan = build_motion_plan(
        video_id=video_id,
        interval=HookInterval(start_seconds=0, end_seconds=1),
        observations=observations,
        audio_context="ignored heuristic audio",
        motion_prompt="Subject holds still, subtle blink, camera locked.",
    )
    assert plan.motionPrompt.startswith("Subject holds still")
    assert plan.confidence == 1.0
    assert plan.motion_timeline == []
    assert plan.audio_context is None
    assert plan.emotion == "undetermined"


def test_motion_plan_without_prompt_has_zero_confidence() -> None:
    video_id = uuid4()
    plan = build_motion_plan(
        video_id=video_id,
        interval=HookInterval(start_seconds=0, end_seconds=1),
        observations=[],
        audio_context=None,
        motion_prompt=None,
    )
    assert plan.confidence == 0.0
    assert plan.motionPrompt is None
