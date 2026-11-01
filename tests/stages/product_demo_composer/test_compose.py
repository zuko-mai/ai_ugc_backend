import shutil
import subprocess
from pathlib import Path

import pytest

from app.stages.product_demo_composer.compose import compose_videos
from app.stages.product_demo_composer.exceptions import InvalidSettingsError
from app.stages.product_demo_composer.probe import probe_bytes, probe_file
from app.stages.product_demo_composer.schemas import ComposeSettings


def _make_clip(
    path: Path,
    *,
    color: str,
    size: str,
    seconds: float,
    fps: int = 30,
    with_audio: bool = True,
) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d={seconds}:r={fps}",
    ]
    if with_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={seconds}:sample_rate=48000",
                "-c:a",
                "aac",
            ]
        )
    else:
        cmd.append("-an")
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not path.exists():
        pytest.skip(f"failed to generate clip: {result.stderr[:300]}")
    return path


@pytest.mark.asyncio
async def test_hard_cut_concat_preserves_sum_duration(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="red", size="540x960", seconds=1.0)
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="540x960", seconds=2.0)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="540x960",
        aspect_ratio="9:16",
        fps=30,
        transition="none",
        transition_duration_seconds=0,
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    assert result.width == 540
    assert result.height == 960
    assert result.duration_seconds == pytest.approx(3.0, abs=0.15)
    assert len(result.segments) == 2
    assert result.segments[0].type == "hook"
    assert result.segments[0].end_seconds == pytest.approx(1.0, abs=0.05)
    assert result.segments[1].type == "product_demo"
    assert result.segments[1].start_seconds == pytest.approx(1.0, abs=0.05)

    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.has_video
    assert out.has_audio
    assert out.duration_seconds == pytest.approx(3.0, abs=0.25)
    assert out.width == 540
    assert out.height == 960


@pytest.mark.asyncio
async def test_scale_blur_fills_vertical_frame(tmp_path: Path) -> None:
    # Landscape into 9:16 with blur fill → exact target size, no black bars.
    hook = _make_clip(tmp_path / "hook.mp4", color="green", size="640x360", seconds=0.5)
    demo = _make_clip(tmp_path / "demo.mp4", color="yellow", size="320x240", seconds=0.5)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="540x960",
        aspect_ratio="9:16",
        fps=24,
        fit_mode="blur",
        transition="none",
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    assert result.width == 540
    assert result.height == 960
    assert ComposeSettings().fit_mode == "blur"
    assert any(w.startswith("hook_scaled:") for w in result.warnings)
    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.width == 540
    assert out.height == 960


@pytest.mark.asyncio
async def test_scale_cover_fills_vertical_frame(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="green", size="640x360", seconds=0.5)
    demo = _make_clip(tmp_path / "demo.mp4", color="yellow", size="320x240", seconds=0.5)
    settings = ComposeSettings(
        resolution="540x960",
        aspect_ratio="9:16",
        fps=24,
        fit_mode="cover",
        transition="none",
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=await probe_file(hook),
        demo_probe=await probe_file(demo),
        settings=settings,
    )
    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.width == 540
    assert out.height == 960


@pytest.mark.asyncio
async def test_scale_contain_letterboxes(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="green", size="640x360", seconds=0.5)
    demo = _make_clip(tmp_path / "demo.mp4", color="yellow", size="320x240", seconds=0.5)
    settings = ComposeSettings(
        resolution="540x960",
        aspect_ratio="9:16",
        fps=24,
        fit_mode="contain",
        transition="none",
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=await probe_file(hook),
        demo_probe=await probe_file(demo),
        settings=settings,
    )
    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.width == 540
    assert out.height == 960


@pytest.mark.asyncio
async def test_crossfade_shortens_total_duration(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="red", size="320x568", seconds=1.0)
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="320x568", seconds=1.0)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="320x568",
        aspect_ratio="9:16",
        fps=30,
        transition="crossfade",
        transition_duration_seconds=0.25,
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    assert result.duration_seconds == pytest.approx(1.75, abs=0.1)
    assert result.segments[1].start_seconds == pytest.approx(0.75, abs=0.05)


@pytest.mark.asyncio
async def test_fade_keeps_full_duration(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="red", size="320x568", seconds=1.0)
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="320x568", seconds=1.0)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="320x568",
        aspect_ratio="9:16",
        fps=30,
        transition="fade",
        transition_duration_seconds=0.2,
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    assert result.duration_seconds == pytest.approx(2.0, abs=0.1)


@pytest.mark.asyncio
async def test_audio_mode_none_still_outputs_track(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="red", size="320x568", seconds=0.5)
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="320x568", seconds=0.5)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="320x568",
        aspect_ratio="9:16",
        audio_mode="none",
    )
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.has_audio  # silent track for pipeline consistency


@pytest.mark.asyncio
async def test_missing_audio_on_source_is_ok(tmp_path: Path) -> None:
    hook = _make_clip(
        tmp_path / "hook.mp4", color="red", size="320x568", seconds=0.5, with_audio=False
    )
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="320x568", seconds=0.5)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    assert not hook_probe.has_audio
    settings = ComposeSettings(resolution="320x568", aspect_ratio="9:16")
    result = await compose_videos(
        hook_bytes=hook.read_bytes(),
        demo_bytes=demo.read_bytes(),
        hook_probe=hook_probe,
        demo_probe=demo_probe,
        settings=settings,
    )
    assert "hook_missing_audio" in result.warnings
    out = await probe_bytes(result.video_bytes, suffix=result.extension)
    assert out.has_video
    assert out.has_audio


@pytest.mark.asyncio
async def test_crossfade_too_long_raises(tmp_path: Path) -> None:
    hook = _make_clip(tmp_path / "hook.mp4", color="red", size="320x568", seconds=0.4)
    demo = _make_clip(tmp_path / "demo.mp4", color="blue", size="320x568", seconds=0.4)
    hook_probe = await probe_file(hook)
    demo_probe = await probe_file(demo)
    settings = ComposeSettings(
        resolution="320x568",
        aspect_ratio="9:16",
        transition="crossfade",
        transition_duration_seconds=0.5,
    )
    with pytest.raises(InvalidSettingsError):
        await compose_videos(
            hook_bytes=hook.read_bytes(),
            demo_bytes=demo.read_bytes(),
            hook_probe=hook_probe,
            demo_probe=demo_probe,
            settings=settings,
        )
