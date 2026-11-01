"""Check that a generated file matches the requested hook video."""

from __future__ import annotations

from dataclasses import dataclass

from app.stages.hook_video_generator.probe import ProbedVideo


@dataclass(frozen=True)
class OutputAssessment:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    container: str | None


def assess_output(
    probed: ProbedVideo,
    *,
    requested_duration: float,
    requested_width: int,
    requested_height: int,
    requested_fps: float,
    min_duration: float,
    max_duration: float,
    duration_tolerance: float,
    strict_resolution: bool = True,
) -> OutputAssessment:
    """Validate container, duration, and frame size of a finished video.

    Kling (and similar i2v APIs) pick pixel size from the first-frame aspect and
    a quality tier (720p/1080p), not from an exact WxH. Pass
    `strict_resolution=False` for those providers: a size mismatch becomes a
    warning, not a hard failure.
    """
    errors: list[str] = []
    warnings: list[str] = []
    container = classify_container(probed.format_name)

    if not probed.has_video:
        errors.append("missing_video_stream")
    if container is None:
        errors.append(f"unsupported_container: {probed.format_name or 'unknown'}")
    if probed.duration_seconds <= 0:
        errors.append("duration_unreadable")
    else:
        if not (min_duration - duration_tolerance) <= probed.duration_seconds <= (
            max_duration + duration_tolerance
        ):
            errors.append(
                "duration_out_of_range: "
                f"measured {probed.duration_seconds:.3f}s, "
                f"allowed {min_duration:.3f}-{max_duration:.3f}s"
            )
        if abs(probed.duration_seconds - requested_duration) > duration_tolerance:
            errors.append(
                "duration_mismatch: "
                f"measured {probed.duration_seconds:.3f}s, "
                f"requested {requested_duration:.3f}s, "
                f"tolerance {duration_tolerance:.3f}s"
            )
    if probed.width < 1 or probed.height < 1:
        errors.append("resolution_unreadable")
    elif probed.width != requested_width or probed.height != requested_height:
        message = (
            "resolution_mismatch: "
            f"measured {probed.width}x{probed.height}, "
            f"requested {requested_width}x{requested_height}"
        )
        if strict_resolution:
            errors.append(message)
        else:
            warnings.append(message)
    if probed.fps <= 0:
        errors.append("fps_unreadable")
    elif abs(probed.fps - requested_fps) > 1.0:
        warnings.append(
            f"fps_drift: measured {probed.fps:.3f}, requested {requested_fps:.3f}"
        )

    return OutputAssessment(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        container=container,
    )


def classify_container(format_name: str | None) -> str | None:
    name = (format_name or "").lower()
    if "mp4" in name or "isom" in name:
        return "mp4"
    if "mov" in name or "quicktime" in name:
        return "mov"
    return None
