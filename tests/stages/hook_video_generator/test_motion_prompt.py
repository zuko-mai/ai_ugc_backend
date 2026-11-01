from app.stages.hook_video_generator.motion_prompt import (
    CAMERA_LOCK,
    IDENTITY_LOCK,
    PERFORMANCE_RULE,
    compile_motion_instructions,
)


def test_timeline_becomes_timed_identity_locked_instructions() -> None:
    instructions = compile_motion_instructions(
        {
            "emotion": "curious",
            "emotion_intensity": 0.7,
            "gaze_direction": "toward camera",
            "facial_expression": "slight smile",
            "motionPrompt": "Subtle nod while keeping eye contact.",
            "motion_timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 1.5,
                    "head_motion": "slight tilt right",
                    "eye_motion": "looks at camera",
                    "mouth_motion": "closed",
                    "body_motion": "still",
                    "emotion": "curious",
                },
                {
                    "start_seconds": 1.5,
                    "end_seconds": 3,
                    "head_motion": "nod",
                    "eye_motion": "blink",
                    "mouth_motion": "small smile",
                    "body_motion": "still",
                    "emotion": "amused",
                },
            ],
        },
        duration_seconds=3,
    )
    assert IDENTITY_LOCK in instructions.prompt
    assert CAMERA_LOCK in instructions.prompt
    assert PERFORMANCE_RULE in instructions.prompt
    assert "exactly 3.00 seconds" in instructions.prompt
    assert "Overall emotion: curious (intensity 0.70)." in instructions.prompt
    assert "Gaze: toward camera." in instructions.prompt
    assert "From 0.00s to 1.50s: head: slight tilt right; eyes: looks at camera" in instructions.prompt
    assert "From 1.50s to 3.00s: head: nod; eyes: blink" in instructions.prompt
    assert "Subtle nod while keeping eye contact." in instructions.prompt
    assert "identity drift" in instructions.negative_prompt
    assert instructions.warnings == ()


def test_empty_plan_holds_identity_without_invented_motion() -> None:
    instructions = compile_motion_instructions({}, duration_seconds=3)
    assert IDENTITY_LOCK in instructions.prompt
    assert CAMERA_LOCK in instructions.prompt
    assert "Do not add head, eye, or body movement." in instructions.prompt
    assert any(item.startswith("motion_plan_empty") for item in instructions.warnings)


def test_cues_outside_the_clip_are_clamped_or_dropped() -> None:
    instructions = compile_motion_instructions(
        {
            "motion_timeline": [
                {"start_seconds": 2.5, "end_seconds": 5, "head_motion": "nod"},
                {"start_seconds": 5, "end_seconds": 6, "head_motion": "shake"},
            ]
        },
        duration_seconds=3,
    )
    assert "From 2.50s to 3.00s: head: nod." in instructions.prompt
    assert "shake" not in instructions.prompt
    assert any(item.startswith("motion_cue_clamped") for item in instructions.warnings)
    assert any(item.startswith("motion_cue_outside") for item in instructions.warnings)


def test_requested_camera_motion_replaces_the_lock() -> None:
    instructions = compile_motion_instructions(
        {"camera_motion": "slow push-in", "emotion": "calm"},
        duration_seconds=2,
    )
    assert "Camera motion is limited to: slow push-in." in instructions.prompt
    assert CAMERA_LOCK not in instructions.prompt
    assert IDENTITY_LOCK in instructions.prompt
    assert any(item.startswith("camera_motion_requested") for item in instructions.warnings)
    assert "other than the requested camera motion" in instructions.negative_prompt
