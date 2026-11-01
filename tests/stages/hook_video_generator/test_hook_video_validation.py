import shutil

import pytest

from app.stages.hook_video_generator.exceptions import ProbeError
from app.stages.hook_video_generator.probe import ProbedVideo, probe_bytes
from app.stages.hook_video_generator.render import render_still_video
from app.stages.hook_video_generator.validation import assess_output


def _probed(**overrides: object) -> ProbedVideo:
    payload = {
        "duration_seconds": 3.0,
        "fps": 30.0,
        "width": 1080,
        "height": 1920,
        "video_codec": "h264",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "has_video": True,
    }
    payload.update(overrides)
    return ProbedVideo(**payload)  # type: ignore[arg-type]


def _assess(probed: ProbedVideo, **overrides: object):
    params = {
        "requested_duration": 3.0,
        "requested_width": 1080,
        "requested_height": 1920,
        "requested_fps": 30.0,
        "min_duration": 2.0,
        "max_duration": 4.0,
        "duration_tolerance": 0.25,
    }
    params.update(overrides)
    return assess_output(probed, **params)  # type: ignore[arg-type]


def test_matching_file_passes() -> None:
    result = _assess(_probed())
    assert result.ok
    assert result.errors == ()
    assert result.container == "mp4"


def test_duration_and_resolution_mismatches_fail() -> None:
    short = _assess(_probed(duration_seconds=1.0))
    assert not short.ok
    assert any(item.startswith("duration_mismatch") for item in short.errors)
    assert any(item.startswith("duration_out_of_range") for item in short.errors)

    resized = _assess(_probed(width=720, height=1280))
    assert not resized.ok
    assert any(item.startswith("resolution_mismatch") for item in resized.errors)


def test_kling_style_resolution_mismatch_is_warning_only() -> None:
    result = _assess(
        _probed(width=1244, height=1660, fps=24.0),
        strict_resolution=False,
    )
    assert result.ok
    assert any(item.startswith("resolution_mismatch") for item in result.warnings)
    assert any(item.startswith("fps_drift") for item in result.warnings)


def test_container_and_stream_are_required() -> None:
    webm = _assess(_probed(format_name="matroska,webm"))
    assert not webm.ok
    assert any(item.startswith("unsupported_container") for item in webm.errors)

    audio_only = _assess(_probed(has_video=False, width=0, height=0, video_codec=None))
    assert not audio_only.ok
    assert "missing_video_stream" in audio_only.errors


def test_fps_drift_is_a_warning() -> None:
    result = _assess(_probed(fps=24.0))
    assert result.ok
    assert any(item.startswith("fps_drift") for item in result.warnings)


def _jpeg() -> bytes:
    import cv2
    import numpy as np

    image = np.full((640, 360, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)
async def test_rendered_file_passes_probe_and_assessment() -> None:
    data = await render_still_video(
        _jpeg(),
        width=360,
        height=640,
        fps=30,
        duration_seconds=2,
    )
    probed = await probe_bytes(data)
    result = assess_output(
        probed,
        requested_duration=2,
        requested_width=360,
        requested_height=640,
        requested_fps=30,
        min_duration=2,
        max_duration=4,
        duration_tolerance=0.25,
    )
    assert result.ok, result.errors
    assert result.container == "mp4"
    assert abs(probed.duration_seconds - 2) <= 0.25
    assert probed.width == 360
    assert probed.height == 640


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
async def test_probe_rejects_non_video_bytes() -> None:
    with pytest.raises(ProbeError):
        await probe_bytes(b"this is not a video")
