import asyncio
import json
import shutil
from fractions import Fraction
from pathlib import Path

from app.core.logging import get_logger
from app.stages.video_ingestion.exceptions import MetadataExtractionError
from app.stages.video_ingestion.schemas import VideoMetadata

logger = get_logger(__name__)


def _ensure_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MetadataExtractionError(
            "ffprobe is not installed or not available on PATH. "
            "Install FFmpeg to enable video metadata extraction."
        )
    return path


def _parse_fps(rate: str | None) -> float:
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        try:
            return float(rate)
        except ValueError:
            return 0.0


async def extract_metadata(file_path: Path) -> VideoMetadata:
    """Extract media metadata via ffprobe without loading the file into memory."""
    ffprobe = _ensure_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    logger.debug("Running ffprobe on %s", file_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except OSError as exc:
        raise MetadataExtractionError(
            "Failed to execute ffprobe",
            details={"error": str(exc)},
        ) from exc

    if proc.returncode != 0:
        raise MetadataExtractionError(
            "ffprobe failed to read media file",
            details={
                "returncode": proc.returncode,
                "stderr": stderr.decode(errors="replace")[:1000],
            },
        )

    try:
        payload = json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise MetadataExtractionError(
            "ffprobe returned invalid JSON",
            details={"error": str(exc)},
        ) from exc

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise MetadataExtractionError(
            "No video stream found in file",
            details={"path": str(file_path)},
        )

    duration = float(fmt.get("duration") or video_stream.get("duration") or 0.0)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    metadata = VideoMetadata(
        duration_seconds=round(duration, 3),
        fps=round(fps, 3),
        width=width,
        height=height,
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        has_audio=audio_stream is not None,
    )
    logger.info(
        "Metadata extracted path=%s duration=%s fps=%s %sx%s video=%s audio=%s",
        file_path.name,
        metadata.duration_seconds,
        metadata.fps,
        metadata.width,
        metadata.height,
        metadata.video_codec,
        metadata.audio_codec,
    )
    return metadata
