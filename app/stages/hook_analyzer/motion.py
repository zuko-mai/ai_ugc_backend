from uuid import UUID

from app.stages.hook_analyzer.schemas import (
    FaceObservation,
    GeminiUsage,
    HookInterval,
    MotionPlanResponse,
)

NOT_OBSERVED = "not clearly observed"


def build_motion_plan(
    *,
    video_id: UUID,
    interval: HookInterval,
    observations: list[FaceObservation],
    audio_context: str | None = None,
    selected_frame_id: UUID | None = None,
    analysis_key: str | None = None,
    motion_prompt: str | None = None,
    trimmed_video_url: str | None = None,
    gemini_usage: GeminiUsage | None = None,
) -> MotionPlanResponse:
    """Return the AI motionPrompt as the primary hook plot.

    Heuristic OpenCV emotion/timeline fields are kept only as empty placeholders
    for backward-compatible response shape. Candidate frames stay separate.
    """
    _ = (observations, audio_context)
    return MotionPlanResponse(
        video_id=video_id,
        interval=interval,
        emotion="undetermined",
        emotion_intensity=0.0,
        gaze_direction=NOT_OBSERVED,
        facial_expression=NOT_OBSERVED,
        motion_timeline=[],
        audio_context=None,
        confidence=1.0 if motion_prompt else 0.0,
        selected_frame_id=selected_frame_id,
        analysis_key=analysis_key,
        motionPrompt=motion_prompt,
        trimmed_video_url=trimmed_video_url,
        gemini_usage=gemini_usage,
    )
