import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.stages.hook_text_engine.exceptions import ExtractionError

logger = get_logger(__name__)


@dataclass
class ExtractedFrame:
    timestamp_seconds: float
    image_bgr: np.ndarray
    path: Path | None = None


async def extract_frames(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    sample_fps: float = 2.0,
    output_dir: Path | None = None,
) -> list[ExtractedFrame]:
    """Sample frames from [start, end) for OCR."""
    if end_seconds <= start_seconds:
        raise ExtractionError("Invalid extraction interval")

    duration = end_seconds - start_seconds
    safe_end = max(start_seconds, end_seconds - 0.05)
    target_count = max(2, min(24, int(duration * sample_fps) + 1))
    if target_count == 1 or safe_end <= start_seconds:
        timestamps = [round(start_seconds, 3)]
    else:
        timestamps = [
            round(
                start_seconds + i * ((safe_end - start_seconds) / (target_count - 1)),
                3,
            )
            for i in range(target_count)
        ]
    timestamps = sorted({max(start_seconds, min(ts, safe_end)) for ts in timestamps})

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[ExtractedFrame] = []
    for index, ts in enumerate(timestamps):
        image = await _read_frame_at(video_path, ts)
        path = None
        if output_dir is not None:
            path = output_dir / f"ocr_{index:03d}_{ts:.3f}.jpg"
            ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise ExtractionError(
                    "Failed to write OCR frame",
                    details={"path": str(path)},
                )
        frames.append(ExtractedFrame(timestamp_seconds=ts, image_bgr=image, path=path))

    logger.info(
        "OCR sampled %s frames from %s [%.3f, %.3f]",
        len(frames),
        video_path.name,
        start_seconds,
        end_seconds,
    )
    return frames


async def _read_frame_at(video_path: Path, timestamp_seconds: float) -> np.ndarray:
    if shutil.which("ffmpeg"):
        try:
            return await _ffmpeg_frame(video_path, timestamp_seconds)
        except ExtractionError:
            logger.warning("ffmpeg frame extract failed, falling back to OpenCV")
    return await asyncio.to_thread(_opencv_frame, video_path, timestamp_seconds)


async def _ffmpeg_frame(video_path: Path, timestamp_seconds: float) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        raise ExtractionError(
            "ffmpeg failed to extract frame",
            details={
                "timestamp": timestamp_seconds,
                "stderr": stderr.decode(errors="replace")[:500],
            },
        )
    image = cv2.imdecode(np.frombuffer(stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ExtractionError("Failed to decode extracted frame")
    return image


def _opencv_frame(video_path: Path, timestamp_seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExtractionError(
            "Cannot open video file",
            details={"path": str(video_path)},
        )
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        ok, frame = capture.read()
        if (not ok or frame is None) and timestamp_seconds > 0:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
                ok, frame = capture.read()
        if not ok or frame is None:
            raise ExtractionError(
                "OpenCV failed to read frame",
                details={"timestamp": timestamp_seconds},
            )
        return frame
    finally:
        capture.release()
