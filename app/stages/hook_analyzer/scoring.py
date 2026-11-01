from dataclasses import dataclass
from uuid import UUID, uuid4

from app.stages.hook_analyzer.frames import ExtractedFrame
from app.stages.hook_analyzer.face import observe_frame
from app.stages.hook_analyzer.schemas import FaceObservation


@dataclass
class RankedFrame:
    frame_id: UUID
    frame: ExtractedFrame
    observation: FaceObservation
    quality_score: float
    reason: str


def observe_all(frames: list[ExtractedFrame]) -> list[FaceObservation]:
    return [observe_frame(frame.image_bgr, frame.timestamp_seconds) for frame in frames]


def rank_frames(
    frames: list[ExtractedFrame],
    observations: list[FaceObservation],
    *,
    candidate_count: int,
) -> list[RankedFrame]:
    if not frames:
        return []

    max_sharpness = max((o.sharpness for o in observations), default=1.0) or 1.0
    scored: list[RankedFrame] = []

    for frame, obs in zip(frames, observations, strict=True):
        score, reason = _score(frame, obs, max_sharpness)
        scored.append(
            RankedFrame(
                frame_id=uuid4(),
                frame=frame,
                observation=obs,
                quality_score=score,
                reason=reason,
            )
        )

    scored.sort(
        key=lambda item: (item.observation.face_detected, item.quality_score),
        reverse=True,
    )

    selected: list[RankedFrame] = []
    for item in scored:
        if any(abs(item.frame.timestamp_seconds - s.frame.timestamp_seconds) < 0.18 for s in selected):
            continue
        selected.append(item)
        if len(selected) >= candidate_count:
            break

    if len(selected) < candidate_count:
        for item in scored:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= candidate_count:
                break

    selected.sort(key=lambda item: item.frame.timestamp_seconds)
    return selected


def _score(
    frame: ExtractedFrame,
    obs: FaceObservation,
    max_sharpness: float,
) -> tuple[float, str]:
    sharpness_norm = min(obs.sharpness / max_sharpness, 1.0) if max_sharpness > 0 else 0.0

    if not obs.face_detected:
        return (
            round(0.15 * sharpness_norm, 3),
            "no face detected; ranked by frame sharpness only",
        )

    height, width = frame.image_bgr.shape[:2]
    size_score = _peak_score(obs.face_area_ratio, ideal=0.18, width=0.14)
    center_score = 0.5
    if obs.face_center is not None:
        nx = obs.face_center[0] / max(width, 1)
        ny = obs.face_center[1] / max(height, 1)
        center_score = float(
            max(0.0, 1.0 - (abs(nx - 0.5) * 1.6 + abs(ny - 0.38) * 1.2))
        )

    clip_penalty = 0.25 if obs.edge_clipped else 0.0
    score = max(
        0.0,
        min(
            1.0,
            0.35 * size_score
            + 0.25 * sharpness_norm
            + 0.30 * center_score
            + 0.10
            - clip_penalty,
        ),
    )

    reasons = ["face visible"]
    if size_score >= 0.7:
        reasons.append("face size suitable for hook")
    elif obs.face_area_ratio < 0.05:
        reasons.append("face relatively small in frame")
    if center_score >= 0.7:
        reasons.append("composition near preferred talking-head framing")
    if obs.edge_clipped:
        reasons.append("face near frame edge (possible crop)")
    if sharpness_norm >= 0.5:
        reasons.append("acceptable sharpness")
    else:
        reasons.append("soft / low-sharpness frame")

    return round(score, 3), "; ".join(reasons)


def _peak_score(value: float, *, ideal: float, width: float) -> float:
    return float(max(0.0, 1.0 - abs(value - ideal) / width))
