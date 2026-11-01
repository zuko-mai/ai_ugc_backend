import asyncio
import json
import shutil
from pathlib import Path

from app.core.logging import get_logger
from app.stages.hook_text_engine.exceptions import ExtractionError

logger = get_logger(__name__)


async def has_audio_stream(video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "a",
        str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except OSError as exc:
        logger.warning("ffprobe audio probe failed: %s", exc)
        return False

    if proc.returncode != 0:
        return False
    try:
        payload = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return False
    return bool(payload.get("streams"))


async def extract_audio_wav(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
) -> Path | None:
    """Extract mono 16kHz WAV for ASR. Returns None if no audio / ffmpeg missing."""
    if not await has_audio_stream(video_path):
        logger.info("No audio stream in %s — skipping ASR", video_path.name)
        return None
    if not shutil.which("ffmpeg"):
        raise ExtractionError("ffmpeg is required to extract audio for ASR")

    duration = max(0.05, end_seconds - start_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        # Silent/corrupt audio — treat as no speech rather than hard fail.
        logger.warning(
            "Audio extract empty/failed: %s",
            stderr.decode(errors="replace")[:400],
        )
        return None
    return output_path
