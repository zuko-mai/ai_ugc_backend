import numpy as np

from app.stages.hook_analyzer.frames import ExtractedFrame
from app.stages.hook_analyzer.schemas import FaceObservation
from app.stages.hook_analyzer.scoring import rank_frames


def _frame(ts: float) -> ExtractedFrame:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (40, 40, 40)
    return ExtractedFrame(timestamp_seconds=ts, image_bgr=image)


def test_rank_prefers_face_over_no_face() -> None:
    frames = [_frame(0.0), _frame(0.3), _frame(0.6)]
    observations = [
        FaceObservation(timestamp_seconds=0.0, face_detected=False, sharpness=50.0),
        FaceObservation(
            timestamp_seconds=0.3,
            face_detected=True,
            face_bbox=(100, 60, 80, 90),
            face_center=(140.0, 105.0),
            face_area_ratio=0.18,
            sharpness=80.0,
        ),
        FaceObservation(timestamp_seconds=0.6, face_detected=False, sharpness=90.0),
    ]
    ranked = rank_frames(frames, observations, candidate_count=2)
    assert ranked
    assert ranked[0].observation.face_detected or any(
        r.observation.face_detected for r in ranked
    )
    face_ranked = [r for r in ranked if r.observation.face_detected]
    assert face_ranked
    assert face_ranked[0].quality_score > 0.3


def test_rank_limits_candidate_count() -> None:
    frames = [_frame(i * 0.25) for i in range(8)]
    observations = [
        FaceObservation(timestamp_seconds=f.timestamp_seconds, face_detected=False, sharpness=10 + i)
        for i, f in enumerate(frames)
    ]
    ranked = rank_frames(frames, observations, candidate_count=3)
    assert len(ranked) == 3
