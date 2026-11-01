from functools import lru_cache

import cv2
import numpy as np

from app.core.logging import get_logger
from app.stages.hook_analyzer.frames import compute_sharpness
from app.stages.hook_analyzer.schemas import FaceObservation

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _face_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {path}")
    return cascade


def observe_frame(image_bgr: np.ndarray, timestamp_seconds: float) -> FaceObservation:
    """Measure only what is visible in the frame (no invented attributes)."""
    height, width = image_bgr.shape[:2]
    sharpness = compute_sharpness(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = _face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(24, width // 20), max(24, height // 20)),
    )

    if len(faces) == 0:
        return FaceObservation(
            timestamp_seconds=timestamp_seconds,
            face_detected=False,
            sharpness=sharpness,
        )

    # Largest face = primary subject for UGC hooks.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    area_ratio = (w * h) / float(width * height)
    cx = x + w / 2.0
    cy = y + h / 2.0
    margin = 0.02
    edge_clipped = (
        x <= width * margin
        or y <= height * margin
        or (x + w) >= width * (1 - margin)
        or (y + h) >= height * (1 - margin)
    )

    mouth_openness = _estimate_mouth_openness(gray, x, y, w, h)
    gaze_x, gaze_y = _estimate_gaze_offset(gray, x, y, w, h)

    return FaceObservation(
        timestamp_seconds=timestamp_seconds,
        face_detected=True,
        face_bbox=(int(x), int(y), int(w), int(h)),
        face_center=(float(cx), float(cy)),
        face_area_ratio=float(area_ratio),
        sharpness=sharpness,
        edge_clipped=bool(edge_clipped),
        mouth_openness=mouth_openness,
        gaze_offset_x=gaze_x,
        gaze_offset_y=gaze_y,
    )


def _estimate_mouth_openness(
    gray: np.ndarray, x: int, y: int, w: int, h: int
) -> float | None:
    """Proxy for mouth openness via contrast in the lower face third.

    Returns a normalized score or None if ROI is invalid.
    """
    mouth_y0 = y + int(h * 0.62)
    mouth_y1 = y + int(h * 0.92)
    mouth_x0 = x + int(w * 0.25)
    mouth_x1 = x + int(w * 0.75)
    if mouth_y1 <= mouth_y0 or mouth_x1 <= mouth_x0:
        return None
    roi = gray[mouth_y0:mouth_y1, mouth_x0:mouth_x1]
    if roi.size == 0:
        return None
    # Higher vertical gradient variance often correlates with open lips / teeth.
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    score = float(np.clip(np.var(sobel_y) / 2500.0, 0.0, 1.0))
    return score


def _estimate_gaze_offset(
    gray: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[float | None, float | None]:
    """Rough gaze proxy from darker pupil mass in the eye band.

    Returns offsets in [-1, 1] relative to eye-band center, or (None, None).
    """
    eye_y0 = y + int(h * 0.28)
    eye_y1 = y + int(h * 0.48)
    eye_x0 = x + int(w * 0.15)
    eye_x1 = x + int(w * 0.85)
    if eye_y1 <= eye_y0 or eye_x1 <= eye_x0:
        return None, None
    roi = gray[eye_y0:eye_y1, eye_x0:eye_x1]
    if roi.size == 0:
        return None, None

    # Darkest 15% of pixels as pupil/iris proxy.
    threshold = np.percentile(roi, 15)
    mask = roi <= threshold
    if not np.any(mask):
        return None, None
    ys, xs = np.where(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    norm_x = (cx / max(roi.shape[1] - 1, 1)) * 2 - 1
    norm_y = (cy / max(roi.shape[0] - 1, 1)) * 2 - 1
    return float(np.clip(norm_x, -1, 1)), float(np.clip(norm_y, -1, 1))
