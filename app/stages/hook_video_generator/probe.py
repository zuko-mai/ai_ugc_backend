"""Read container, duration, and frame size from a generated video file."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.stages.hook_video_generator.exceptions import ProbeError


@dataclass(frozen=True)
class ProbedVideo:
    duration_seconds: float
    fps: float
    width: int
    height: int
    video_codec: str | None
    format_name: str | None
    has_video: bool


async def probe_bytes(data: bytes, *, suffix: str = ".mp4") -> ProbedVideo:
    path = await asyncio.to_thread(_write_temp, data, suffix)
    try:
        return await probe_file(path)
    finally:
        path.unlink(missing_ok=True)


async def probe_file(file_path: Path) -> ProbedVideo:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ProbeError(
            "ffprobe is not installed or not available on PATH. "
            "Install FFmpeg to validate generated videos."
        )
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
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError as exc:
        raise ProbeError("ffprobe timed out while reading the generated video") from exc
    except OSError as exc:
        raise ProbeError("Failed to execute ffprobe", details={"error": str(exc)}) from exc

    if proc.returncode != 0:
        raise ProbeError(
            "ffprobe could not read the generated file",
            details={"stderr": stderr.decode(errors="replace")[:800]},
        )
    try:
        payload = json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    duration = float(fmt.get("duration") or (video_stream or {}).get("duration") or 0.0)
    width = int((video_stream or {}).get("width") or 0)
    height = int((video_stream or {}).get("height") or 0)
    fps = _parse_fps(
        (video_stream or {}).get("avg_frame_rate") or (video_stream or {}).get("r_frame_rate")
    )
    return ProbedVideo(
        duration_seconds=duration,
        fps=fps,
        width=width,
        height=height,
        video_codec=(video_stream or {}).get("codec_name"),
        format_name=fmt.get("format_name"),
        has_video=video_stream is not None,
    )


def _write_temp(data: bytes, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(data)
    finally:
        handle.close()
    return Path(handle.name)


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
