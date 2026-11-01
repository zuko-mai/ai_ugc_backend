import asyncio
import shutil
from pathlib import Path

from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import FrameExtractionError

logger = get_logger(__name__)


async def trim_video_clip(
    source_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    output_path: Path,
) -> Path:
    """Cut [start, start+duration) with ffmpeg stream copy when possible."""
    if duration_seconds <= 0:
        raise FrameExtractionError("duration_seconds must be > 0")
    if not shutil.which("ffmpeg"):
        raise FrameExtractionError("ffmpeg is required to trim hook clips")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode short hooks for reliable cuts (stream copy often keyframe-snaps).
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise FrameExtractionError(
            "Failed to trim video clip",
            details={
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "stderr": stderr.decode(errors="replace")[:800],
            },
        )
    logger.info(
        "Trimmed clip start=%.3f duration=%.3f -> %s (%s bytes)",
        start_seconds,
        duration_seconds,
        output_path.name,
        output_path.stat().st_size,
    )
    return output_path
