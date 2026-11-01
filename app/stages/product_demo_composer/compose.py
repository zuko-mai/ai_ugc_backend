"""FFmpeg composition: normalize both clips, then [hook] → [demo]."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.stages.product_demo_composer.exceptions import ComposeError, InvalidSettingsError
from app.stages.product_demo_composer.probe import ProbedVideo
from app.stages.product_demo_composer.schemas import (
    ComposeSettings,
    TimelineSegment,
    parse_resolution,
)

_AUDIO_SAMPLE_RATE = 48000
_AUDIO_CHANNELS = 2


@dataclass(frozen=True)
class ComposeResult:
    video_bytes: bytes
    extension: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    segments: list[TimelineSegment]
    warnings: list[str]


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ComposeError("ffmpeg is not installed or not available on PATH")
    return ffmpeg


def _video_norm_chain(
    input_label: str,
    output_label: str,
    width: int,
    height: int,
    fps: int,
    *,
    fit_mode: str,
) -> str:
    """Normalize one stream to target frame without stretch.

    blur    — full content centered; empty areas filled with blurred same clip
    cover   — scale up + center-crop
    contain — scale down + black letterbox/pillarbox
    """
    tag = output_label  # unique suffix for intermediate labels
    if fit_mode == "blur":
        return (
            f"[{input_label}]split=2[bg{tag}][fg{tag}];"
            f"[bg{tag}]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=28:steps=2,eq=saturation=0.9"
            f"[bgout{tag}];"
            f"[fg{tag}]scale={width}:{height}:force_original_aspect_ratio=decrease"
            f"[fgout{tag}];"
            f"[bgout{tag}][fgout{tag}]overlay=(W-w)/2:(H-h)/2,"
            f"setsar=1,fps={fps},format=yuv420p[{output_label}]"
        )
    if fit_mode == "contain":
        geometry = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        geometry = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    return (
        f"[{input_label}]{geometry},setsar=1,fps={fps},format=yuv420p[{output_label}]"
    )


def _audio_keep_filter() -> str:
    return (
        f"aformat=sample_rates={_AUDIO_SAMPLE_RATE}:channel_layouts=stereo,"
        "aresample=async=1:first_pts=0"
    )


def _build_timeline(
    *,
    hook_duration: float,
    demo_duration: float,
    transition: str,
    transition_duration: float,
) -> tuple[list[TimelineSegment], float]:
    if transition == "crossfade":
        total = hook_duration + demo_duration - transition_duration
        hook_end = hook_duration
        demo_start = hook_duration - transition_duration
    else:
        # none and fade: full sequential durations (fade does not shorten timeline)
        total = hook_duration + demo_duration
        hook_end = hook_duration
        demo_start = hook_duration

    segments = [
        TimelineSegment(type="hook", start_seconds=0.0, end_seconds=round(hook_end, 3)),
        TimelineSegment(
            type="product_demo",
            start_seconds=round(demo_start, 3),
            end_seconds=round(total, 3),
        ),
    ]
    return segments, total


def _validate_durations(
    hook: ProbedVideo,
    demo: ProbedVideo,
    settings: ComposeSettings,
) -> list[str]:
    warnings: list[str] = []
    if hook.duration_seconds <= 0:
        raise InvalidSettingsError(
            "Hook video duration is zero or unknown",
            details={"duration_seconds": hook.duration_seconds},
        )
    if demo.duration_seconds <= 0:
        raise InvalidSettingsError(
            "Demo video duration is zero or unknown",
            details={"duration_seconds": demo.duration_seconds},
        )

    td = settings.transition_duration_seconds
    if settings.transition != "none":
        min_dur = min(hook.duration_seconds, demo.duration_seconds)
        if td >= min_dur:
            raise InvalidSettingsError(
                "transition_duration_seconds must be shorter than both clip durations",
                details={
                    "transition_duration_seconds": td,
                    "hook_duration_seconds": hook.duration_seconds,
                    "demo_duration_seconds": demo.duration_seconds,
                },
            )

    if not hook.has_audio:
        warnings.append("hook_missing_audio")
    if not demo.has_audio:
        warnings.append("demo_missing_audio")

    target_w, target_h = parse_resolution(settings.resolution)
    if (hook.width, hook.height) != (target_w, target_h):
        warnings.append(
            f"hook_scaled:{hook.width}x{hook.height}->{target_w}x{target_h}"
        )
    if (demo.width, demo.height) != (target_w, target_h):
        warnings.append(
            f"demo_scaled:{demo.width}x{demo.height}->{target_w}x{target_h}"
        )
    if hook.fps and abs(hook.fps - settings.fps) > 0.5:
        warnings.append(f"hook_fps_resampled:{hook.fps:.3f}->{settings.fps}")
    if demo.fps and abs(demo.fps - settings.fps) > 0.5:
        warnings.append(f"demo_fps_resampled:{demo.fps:.3f}->{settings.fps}")
    return warnings


def _want_audio(clip: str, settings: ComposeSettings, probed: ProbedVideo) -> bool:
    if settings.audio_mode == "none":
        return False
    if settings.audio_mode == "hook" and clip != "hook":
        return False
    if settings.audio_mode == "demo" and clip != "demo":
        return False
    return probed.has_audio


async def compose_videos(
    *,
    hook_bytes: bytes,
    demo_bytes: bytes,
    hook_probe: ProbedVideo,
    demo_probe: ProbedVideo,
    settings: ComposeSettings,
    hook_suffix: str = ".mp4",
    demo_suffix: str = ".mp4",
) -> ComposeResult:
    """Professionally join hook then demo. Never mutates the source bytes."""
    ffmpeg = _require_ffmpeg()
    warnings = _validate_durations(hook_probe, demo_probe, settings)
    width, height = parse_resolution(settings.resolution)
    ext = ".mov" if settings.output_format == "mov" else ".mp4"

    segments, expected_duration = _build_timeline(
        hook_duration=hook_probe.duration_seconds,
        demo_duration=demo_probe.duration_seconds,
        transition=settings.transition,
        transition_duration=settings.transition_duration_seconds,
    )

    hook_audio = _want_audio("hook", settings, hook_probe)
    demo_audio = _want_audio("demo", settings, demo_probe)

    def _run() -> bytes:
        with tempfile.TemporaryDirectory(prefix="pdc-compose-") as tmp:
            tmp_path = Path(tmp)
            hook_path = tmp_path / f"hook{hook_suffix}"
            demo_path = tmp_path / f"demo{demo_suffix}"
            out_path = tmp_path / f"out{ext}"
            hook_path.write_bytes(hook_bytes)
            demo_path.write_bytes(demo_bytes)

            filter_complex, map_args = _build_filtergraph(
                hook_duration=hook_probe.duration_seconds,
                demo_duration=demo_probe.duration_seconds,
                settings=settings,
                width=width,
                height=height,
                hook_audio=hook_audio,
                demo_audio=demo_audio,
                hook_has_audio=hook_probe.has_audio,
                demo_has_audio=demo_probe.has_audio,
            )

            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(hook_path),
                "-i",
                str(demo_path),
            ]
            # Silence generators when a clip has no audio stream but we need one.
            if not hook_probe.has_audio or not hook_audio:
                cmd.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        (
                            f"anullsrc=channel_layout=stereo:"
                            f"sample_rate={_AUDIO_SAMPLE_RATE}"
                        ),
                    ]
                )
            if not demo_probe.has_audio or not demo_audio:
                cmd.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        (
                            f"anullsrc=channel_layout=stereo:"
                            f"sample_rate={_AUDIO_SAMPLE_RATE}"
                        ),
                    ]
                )

            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    *map_args,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-t",
                    f"{expected_duration:.6f}",
                    str(out_path),
                ]
            )

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
                raise ComposeError(
                    "ffmpeg failed to compose hook and demo videos",
                    details={"stderr": (result.stderr or "")[:1200]},
                )
            return out_path.read_bytes()

    video_bytes = await asyncio.to_thread(_run)
    return ComposeResult(
        video_bytes=video_bytes,
        extension=ext,
        duration_seconds=round(expected_duration, 3),
        width=width,
        height=height,
        fps=float(settings.fps),
        segments=segments,
        warnings=warnings,
    )


def _build_filtergraph(
    *,
    hook_duration: float,
    demo_duration: float,
    settings: ComposeSettings,
    width: int,
    height: int,
    hook_audio: bool,
    demo_audio: bool,
    hook_has_audio: bool,
    demo_has_audio: bool,
) -> tuple[str, list[str]]:
    """Build filter_complex and -map args.

    Input indices:
      0 = hook video file
      1 = demo video file
      optional lavfi anullsrc inputs appended when silence is needed
    """
    anorm = _audio_keep_filter()
    td = settings.transition_duration_seconds

    # Assign silence input indices after the two video files.
    next_idx = 2
    hook_silence_idx: int | None = None
    demo_silence_idx: int | None = None
    if not hook_has_audio or not hook_audio:
        hook_silence_idx = next_idx
        next_idx += 1
    if not demo_has_audio or not demo_audio:
        demo_silence_idx = next_idx
        next_idx += 1

    parts: list[str] = [
        _video_norm_chain(
            "0:v",
            "v0",
            width,
            height,
            settings.fps,
            fit_mode=settings.fit_mode,
        ),
        _video_norm_chain(
            "1:v",
            "v1",
            width,
            height,
            settings.fps,
            fit_mode=settings.fit_mode,
        ),
    ]

    # Audio labels [a0] / [a1] always exist after this block.
    if hook_audio and hook_has_audio:
        parts.append(f"[0:a]{anorm}[a0]")
    else:
        assert hook_silence_idx is not None
        parts.append(
            f"[{hook_silence_idx}:a]atrim=0:{hook_duration:.6f},"
            f"asetpts=PTS-STARTPTS,{anorm}[a0]"
        )

    if demo_audio and demo_has_audio:
        parts.append(f"[1:a]{anorm}[a1]")
    else:
        assert demo_silence_idx is not None
        parts.append(
            f"[{demo_silence_idx}:a]atrim=0:{demo_duration:.6f},"
            f"asetpts=PTS-STARTPTS,{anorm}[a1]"
        )

    if settings.transition == "none":
        parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]")
    elif settings.transition == "fade":
        fade_out_start = max(0.0, hook_duration - td)
        parts.append(
            f"[v0]fade=t=out:st={fade_out_start:.6f}:d={td:.6f}[v0f]"
        )
        parts.append(f"[v1]fade=t=in:st=0:d={td:.6f}[v1f]")
        parts.append(
            f"[a0]afade=t=out:st={fade_out_start:.6f}:d={td:.6f}[a0f]"
        )
        parts.append(f"[a1]afade=t=in:st=0:d={td:.6f}[a1f]")
        parts.append("[v0f][a0f][v1f][a1f]concat=n=2:v=1:a=1[outv][outa]")
    elif settings.transition == "crossfade":
        offset = max(0.0, hook_duration - td)
        parts.append(
            f"[v0][v1]xfade=transition=fade:duration={td:.6f}:offset={offset:.6f}[outv]"
        )
        parts.append(f"[a0][a1]acrossfade=d={td:.6f}[outa]")
    else:
        raise InvalidSettingsError(
            f"Unsupported transition '{settings.transition}'",
            details={"transition": settings.transition},
        )

    filter_complex = ";".join(parts)
    map_args = ["-map", "[outv]", "-map", "[outa]"]
    return filter_complex, map_args
