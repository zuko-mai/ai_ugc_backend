"""FFmpeg music bed: trim, loop, delay, fade, level, and mix without touching video.

`silence` and `trim` both play the excerpt once and leave the rest of the video
without music. The video timeline is never shortened. `loop` repeats the excerpt
that starts at `music_start_offset_seconds` until the video window is filled.
When the excerpt is longer than the remaining video, every mode cuts it to fit.

Source files are only read. Encoding writes a new file.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.stages.music_overlay.exceptions import InvalidSettingsError, OverlayError
from app.stages.music_overlay.probe import ProbedMedia
from app.stages.music_overlay.schemas import OverlaySettings

logger = get_logger(__name__)

SAMPLE_RATE = 48000
LIMITER_LINEAR = 0.95
_MIN_PLAY_SECONDS = 0.05
_COVERS_WINDOW_SLACK = 0.02


@dataclass(frozen=True)
class OverlayPlan:
    """Resolved timeline. Offsets stay independent of each other."""

    music_start_offset_seconds: float
    video_start_seconds: float
    available_music_seconds: float
    playback_window_seconds: float
    music_play_seconds: float
    loop: bool
    fade_in_seconds: float
    fade_out_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OverlayResult:
    video_bytes: bytes
    extension: str
    duration_seconds: float
    warnings: list[str]
    plan: OverlayPlan


def build_plan(
    *,
    video_duration: float,
    music_duration: float,
    settings: OverlaySettings,
) -> OverlayPlan:
    """Validate offsets against real durations and decide loop vs one-shot."""
    if not math.isfinite(video_duration) or video_duration <= 0:
        raise InvalidSettingsError(
            "Video duration is zero or unknown",
            details={"duration_seconds": video_duration},
        )
    if not math.isfinite(music_duration) or music_duration <= 0:
        raise InvalidSettingsError(
            "Music duration is zero or unknown",
            details={"duration_seconds": music_duration},
        )

    offset = settings.music_start_offset_seconds
    video_start = settings.video_start_seconds
    if offset >= music_duration - 1e-3:
        raise InvalidSettingsError(
            "music_start_offset_seconds exceeds the music file duration",
            details={
                "music_start_offset_seconds": offset,
                "music_duration_seconds": round(music_duration, 3),
            },
        )
    if video_start >= video_duration - 1e-3:
        raise InvalidSettingsError(
            "video_start_seconds exceeds the video duration",
            details={
                "video_start_seconds": video_start,
                "video_duration_seconds": round(video_duration, 3),
            },
        )

    available = music_duration - offset
    window = video_duration - video_start
    if available < _MIN_PLAY_SECONDS:
        raise InvalidSettingsError(
            "music_start_offset_seconds leaves no usable music",
            details={
                "music_start_offset_seconds": offset,
                "available_music_seconds": round(available, 3),
            },
        )
    if window < _MIN_PLAY_SECONDS:
        raise InvalidSettingsError(
            "video_start_seconds leaves no room for music on the video",
            details={
                "video_start_seconds": video_start,
                "playback_window_seconds": round(window, 3),
            },
        )

    warnings: list[str] = []
    behavior = settings.music_end_behavior
    if available + _COVERS_WINDOW_SLACK < window:
        if behavior == "loop":
            music_play = window
            do_loop = True
            warnings.append("music_looped_to_fill_video")
        elif behavior == "silence":
            music_play = available
            do_loop = False
            warnings.append("music_ended_before_video")
        else:
            music_play = available
            do_loop = False
            warnings.append("music_shorter_than_video_no_loop")
    else:
        music_play = window
        do_loop = False
        if available > window + _COVERS_WINDOW_SLACK:
            warnings.append("music_trimmed_to_video")
        if behavior == "loop":
            warnings.append("loop_not_needed_music_covers_window")

    fade_in = settings.fade_in_seconds
    fade_out = settings.fade_out_seconds
    if fade_in + fade_out > music_play + 1e-6:
        raise InvalidSettingsError(
            "fade_in_seconds + fade_out_seconds exceeds the music playback duration",
            details={
                "fade_in_seconds": fade_in,
                "fade_out_seconds": fade_out,
                "music_play_seconds": round(music_play, 3),
            },
        )

    return OverlayPlan(
        music_start_offset_seconds=offset,
        video_start_seconds=video_start,
        available_music_seconds=available,
        playback_window_seconds=window,
        music_play_seconds=music_play,
        loop=do_loop,
        fade_in_seconds=fade_in,
        fade_out_seconds=fade_out,
        warnings=warnings,
    )


def build_extract_filter(offset_seconds: float) -> str:
    """Decode from the music offset at a common rate and channel layout."""
    return (
        f"aresample={SAMPLE_RATE}:async=1:first_pts=0,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"atrim=start={offset_seconds:.6f},"
        "asetpts=PTS-STARTPTS"
    )


def build_bed_filter(plan: OverlayPlan, music_volume: float) -> str:
    """Level, fade, and cut the excerpt. Timeline starts at 0, before video delay."""
    parts = [f"volume={music_volume:.6f}"]
    if plan.fade_in_seconds > 0:
        parts.append(f"afade=t=in:st=0:d={plan.fade_in_seconds:.6f}")
    if plan.fade_out_seconds > 0:
        fade_start = max(0.0, plan.music_play_seconds - plan.fade_out_seconds)
        parts.append(
            f"afade=t=out:st={fade_start:.6f}:d={plan.fade_out_seconds:.6f}"
        )
    parts.append(f"atrim=end={plan.music_play_seconds:.6f}")
    parts.append("asetpts=PTS-STARTPTS")
    return ",".join(parts)


def build_mix_filter(
    *,
    video_duration: float,
    video_start_seconds: float,
    include_original: bool,
    original_volume: float,
) -> str:
    """Place the bed on the video clock and mix. Limiter runs after the sum.

    `normalize=0` keeps the requested linear gains. `alimiter` with `level=false`
    only pulls peaks down to LIMITER_LINEAR and does not boost quiet audio.
    """
    delay_ms = int(round(video_start_seconds * 1000))
    music_chain = ["[1:a]"]
    if delay_ms > 0:
        music_chain.append(f"adelay=delays={delay_ms}:all=1,")
    music_chain.append(
        f"apad=whole_dur={video_duration:.6f},"
        f"atrim=end={video_duration:.6f},"
        "asetpts=PTS-STARTPTS[music]"
    )
    parts = ["".join(music_chain)]

    if include_original:
        parts.append(
            "[0:a]"
            f"aresample={SAMPLE_RATE}:async=1:first_pts=0,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={original_volume:.6f},"
            f"apad=whole_dur={video_duration:.6f},"
            f"atrim=end={video_duration:.6f},"
            "asetpts=PTS-STARTPTS[orig]"
        )
        parts.append(
            "[orig][music]"
            "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"alimiter=limit={LIMITER_LINEAR}:level=false:latency=true,"
            f"atrim=end={video_duration:.6f},"
            "asetpts=PTS-STARTPTS[aout]"
        )
    else:
        parts.append(
            "[music]"
            f"alimiter=limit={LIMITER_LINEAR}:level=false:latency=true,"
            f"atrim=end={video_duration:.6f},"
            "asetpts=PTS-STARTPTS[aout]"
        )
    return ";".join(parts)


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise OverlayError("ffmpeg is not installed or not available on PATH")
    return ffmpeg


def _run_ffmpeg(cmd: list[str], *, timeout: float, what: str) -> None:
    logger.debug("ffmpeg %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OverlayError(
            f"ffmpeg timed out while {what}",
            details={"timeout_seconds": timeout},
        ) from exc
    except OSError as exc:
        raise OverlayError(
            f"Failed to execute ffmpeg while {what}",
            details={"error": str(exc)},
        ) from exc
    if result.returncode != 0:
        raise OverlayError(
            f"ffmpeg failed while {what}",
            details={"stderr": (result.stderr or "")[:1200]},
        )


def _render(
    *,
    video_bytes: bytes,
    music_bytes: bytes,
    video_suffix: str,
    music_suffix: str,
    settings: OverlaySettings,
    plan: OverlayPlan,
    video_duration: float,
    include_original: bool,
    extension: str,
) -> bytes:
    ffmpeg = _require_ffmpeg()
    timeout = max(90.0, video_duration * 20 + plan.available_music_seconds * 2 + 30)
    with tempfile.TemporaryDirectory(prefix="music-overlay-") as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / f"video{video_suffix if video_suffix.startswith('.') else '.mp4'}"
        music_path = tmp_path / f"music{music_suffix if music_suffix.startswith('.') else '.wav'}"
        segment_path = tmp_path / "segment.wav"
        bed_path = tmp_path / "bed.wav"
        out_path = tmp_path / f"out{extension}"
        video_path.write_bytes(video_bytes)
        music_path.write_bytes(music_bytes)

        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(music_path),
                "-vn",
                "-af",
                build_extract_filter(plan.music_start_offset_seconds),
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                str(segment_path),
            ],
            timeout=timeout,
            what="extracting the music excerpt",
        )
        if not segment_path.is_file() or segment_path.stat().st_size == 0:
            raise OverlayError("Music excerpt extraction produced an empty file")

        shape_cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if plan.loop:
            shape_cmd.extend(["-stream_loop", "-1"])
        shape_cmd.extend(
            [
                "-i",
                str(segment_path),
                "-af",
                build_bed_filter(plan, settings.music_volume),
                "-t",
                f"{plan.music_play_seconds:.6f}",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                str(bed_path),
            ]
        )
        _run_ffmpeg(shape_cmd, timeout=timeout, what="shaping the music bed")
        if not bed_path.is_file() or bed_path.stat().st_size == 0:
            raise OverlayError("Music bed shaping produced an empty file")

        graph = build_mix_filter(
            video_duration=video_duration,
            video_start_seconds=plan.video_start_seconds,
            include_original=include_original,
            original_volume=settings.original_audio_volume,
        )
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-i",
                str(bed_path),
                "-filter_complex",
                graph,
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            timeout=timeout,
            what="mixing music onto the video",
        )
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise OverlayError("ffmpeg produced an empty video")
        return out_path.read_bytes()


def _media_warnings(
    *,
    settings: OverlaySettings,
    video_probe: ProbedMedia,
    music_probe: ProbedMedia,
) -> list[str]:
    warnings: list[str] = []
    if settings.original_audio_enabled and not video_probe.has_audio:
        warnings.append("original_audio_missing")
    if music_probe.has_video:
        warnings.append("music_video_stream_ignored")
    if music_probe.sample_rate and music_probe.sample_rate != SAMPLE_RATE:
        warnings.append(f"music_resampled:{music_probe.sample_rate}->{SAMPLE_RATE}")
    if music_probe.channels == 1:
        warnings.append("music_upmixed_to_stereo")
    if (
        settings.original_audio_enabled
        and video_probe.has_audio
        and video_probe.sample_rate
        and video_probe.sample_rate != SAMPLE_RATE
    ):
        warnings.append(
            f"original_audio_resampled:{video_probe.sample_rate}->{SAMPLE_RATE}"
        )
    if settings.music_volume == 0 and (
        not settings.original_audio_enabled
        or not video_probe.has_audio
        or settings.original_audio_volume == 0
    ):
        warnings.append("output_audio_silent")
    return warnings


async def overlay_music(
    *,
    video_bytes: bytes,
    music_bytes: bytes,
    video_probe: ProbedMedia,
    music_probe: ProbedMedia,
    settings: OverlaySettings,
    video_suffix: str = ".mp4",
    music_suffix: str = ".wav",
) -> OverlayResult:
    """Mix music onto a copy of the video. Input bytes are not modified."""
    plan = build_plan(
        video_duration=video_probe.duration_seconds,
        music_duration=music_probe.duration_seconds,
        settings=settings,
    )
    include_original = settings.original_audio_enabled and video_probe.has_audio
    extension = ".mov" if settings.output_format == "mov" else ".mp4"
    warnings = list(plan.warnings) + _media_warnings(
        settings=settings,
        video_probe=video_probe,
        music_probe=music_probe,
    )

    video_out = await asyncio.to_thread(
        _render,
        video_bytes=video_bytes,
        music_bytes=music_bytes,
        video_suffix=video_suffix,
        music_suffix=music_suffix,
        settings=settings,
        plan=plan,
        video_duration=video_probe.duration_seconds,
        include_original=include_original,
        extension=extension,
    )
    logger.info(
        "Music overlay rendered duration=%ss offset=%ss video_start=%ss "
        "play=%ss loop=%s volume=%s",
        round(video_probe.duration_seconds, 3),
        plan.music_start_offset_seconds,
        plan.video_start_seconds,
        round(plan.music_play_seconds, 3),
        plan.loop,
        settings.music_volume,
    )
    return OverlayResult(
        video_bytes=video_out,
        extension=extension,
        duration_seconds=video_probe.duration_seconds,
        warnings=warnings,
        plan=plan,
    )
