"""Normalize provider output to the requested social frame (cover crop)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.stages.hook_video_generator.exceptions import ProbeError


async def normalize_video_cover(
    video_bytes: bytes,
    *,
    width: int,
    height: int,
    fps: int,
    source_suffix: str = ".mp4",
    has_audio: bool = False,
) -> bytes:
    """Scale+center-crop to exact WxH (9:16 Reels/TikTok). No letterbox bars."""
    if width % 2 or height % 2:
        raise ProbeError("normalize width/height must be even")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProbeError("ffmpeg is required to normalize hook video")

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )
    suffix = source_suffix if source_suffix.startswith(".") else f".{source_suffix}"

    def _run() -> bytes:
        with tempfile.TemporaryDirectory(prefix="hookvid-norm-") as tmp:
            src = Path(tmp) / f"in{suffix}"
            dst = Path(tmp) / "out.mp4"
            src.write_bytes(video_bytes)
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
            if has_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            else:
                cmd.append("-an")
            cmd.append(str(dst))
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                stderr = result.stderr.decode(errors="replace")[:800]
                raise ProbeError(f"ffmpeg failed to normalize hook video: {stderr}")
            return dst.read_bytes()

    return await asyncio.to_thread(_run)
