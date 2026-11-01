import asyncio
import json
import shutil
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


async def probe_audio_context(video_path: Path) -> str | None:
    """Return a factual audio note or None — no invented speech content."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

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
        return None

    if proc.returncode != 0:
        return None

    try:
        payload = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    if not streams:
        return "no audio stream detected"

    codec = streams[0].get("codec_name") or "unknown"
    channels = streams[0].get("channels")
    channel_note = f", {channels} ch" if channels else ""
    return f"audio stream present ({codec}{channel_note}); content not transcribed"
