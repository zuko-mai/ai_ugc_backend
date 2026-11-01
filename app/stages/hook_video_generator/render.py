"""Render a still-image video. Used by the local stub provider and tests."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path


class RenderError(Exception):
    """Raised when ffmpeg cannot build a video from a still frame."""


async def render_still_video(
    image_bytes: bytes,
    *,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    image_suffix: str = ".jpg",
) -> bytes:
    if width % 2 or height % 2:
        raise RenderError("width and height must be even")
    if fps < 1:
        raise RenderError("fps must be positive")
    if duration_seconds <= 0:
        raise RenderError("duration must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RenderError("ffmpeg is not installed or not available on PATH")

    frames = max(1, round(duration_seconds * fps))
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,format=yuv420p"
    )
    suffix = image_suffix if image_suffix.startswith(".") else f".{image_suffix}"

    def _run() -> bytes:
        with tempfile.TemporaryDirectory(prefix="hookvid-") as tmp:
            src = Path(tmp) / f"frame{suffix}"
            dst = Path(tmp) / "out.mp4"
            src.write_bytes(image_bytes)
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-i",
                str(src),
                "-frames:v",
                str(frames),
                "-vf",
                vf,
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                str(dst),
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                stderr = result.stderr.decode(errors="replace")[:800]
                raise RenderError(f"ffmpeg failed to render the still video: {stderr}")
            return dst.read_bytes()

    return await asyncio.to_thread(_run)
