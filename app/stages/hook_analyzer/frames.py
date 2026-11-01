import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import FrameExtractionError

logger = get_logger(__name__)


@dataclass
class ExtractedFrame:
    timestamp_seconds: float
    image_bgr: np.ndarray
    path: Path | None = None


def compute_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


async def extract_frames(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    sample_fps: float = 8.0,
    output_dir: Path | None = None,
) -> list[ExtractedFrame]:
    """Extract representative frames from [start, end) without loading whole video into RAM."""
    if end_seconds <= start_seconds:
        raise FrameExtractionError("Invalid extraction interval")

    duration = end_seconds - start_seconds
    # Stay slightly before end_seconds — EOF seeks often fail on short clips.
    safe_end = max(start_seconds, end_seconds - 0.05)
    target_count = max(3, min(48, int(duration * sample_fps) + 1))
    if target_count == 1 or safe_end <= start_seconds:
        timestamps = [round(start_seconds, 3)]
    else:
        timestamps = [
            round(start_seconds + i * ((safe_end - start_seconds) / (target_count - 1)), 3)
            for i in range(target_count)
        ]
    timestamps = sorted(set(max(start_seconds, min(ts, safe_end)) for ts in timestamps))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[ExtractedFrame] = []
    for index, ts in enumerate(timestamps):
        image = await _read_frame_at(video_path, ts)
        path = None
        if output_dir is not None:
            path = output_dir / f"sample_{index:03d}_{ts:.3f}.jpg"
            ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise FrameExtractionError(
                    "Failed to write extracted frame",
                    details={"path": str(path)},
                )
        frames.append(ExtractedFrame(timestamp_seconds=ts, image_bgr=image, path=path))

    logger.info(
        "Extracted %s frames from %s [%.3f, %.3f]",
        len(frames),
        video_path.name,
        start_seconds,
        end_seconds,
    )
    return frames


async def _read_frame_at(video_path: Path, timestamp_seconds: float) -> np.ndarray:
    """Seek to timestamp via ffmpeg pipe (stream-friendly) with OpenCV fallback."""
    if shutil.which("ffmpeg"):
        try:
            return await _ffmpeg_frame(video_path, timestamp_seconds)
        except FrameExtractionError:
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
        raise FrameExtractionError(
            "ffmpeg failed to extract frame",
            details={
                "timestamp": timestamp_seconds,
                "stderr": stderr.decode(errors="replace")[:500],
            },
        )
    image = cv2.imdecode(np.frombuffer(stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FrameExtractionError("Failed to decode extracted frame")
    return image


def _opencv_frame(video_path: Path, timestamp_seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameExtractionError(
            "Cannot open video file",
            details={"path": str(video_path)},
        )
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        ok, frame = capture.read()
        if (not ok or frame is None) and timestamp_seconds > 0:
            # Near EOF: grab the last decodable frame.
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
                ok, frame = capture.read()
        if not ok or frame is None:
            raise FrameExtractionError(
                "OpenCV failed to read frame",
                details={"timestamp": timestamp_seconds},
            )
        return frame
    finally:
        capture.release()
