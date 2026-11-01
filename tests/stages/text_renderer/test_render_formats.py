import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.stages.text_renderer.ass import prepare_segments
from app.stages.text_renderer.probe import probe_bytes
from app.stages.text_renderer.render import burn_text_video, export_without_text
from app.stages.text_renderer.schemas import TextSegment, TextStyle
from app.stages.text_renderer.service import TextRendererService
from app.storage.local import LocalFileStorage


def _make_vertical_mp4(path: Path, *, seconds: float = 3.0) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=1080x1920:d={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"failed to make vertical mp4: {result.stderr[:300]}")


@pytest.mark.asyncio
async def test_burn_text_renders_emoji_glyphs_not_tofu(tmp_path: Path) -> None:
    """Noto Emoji fallback should draw silhouettes instead of hollow boxes."""
    src = tmp_path / "in.mp4"
    _make_vertical_mp4(src, seconds=1.0)
    style = TextStyle(font_size=64, padding_top=8, outline_width=2)
    prepared = prepare_segments(
        text="Wait 🔥 for it 😍",
        segments=[],
        style=style,
        frame_width=1080,
        duration_seconds=1.0,
    )
    rendered = await burn_text_video(
        src.read_bytes(),
        style=style,
        segments=prepared,
        width=1080,
        height=1920,
        output_format="mp4",
        preview_at_seconds=0.2,
        source_suffix=".mp4",
    )
    assert rendered.preview_jpeg is not None
    # Tofu boxes are thin rectangles; emoji silhouettes fill more white pixels.
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(rendered.preview_jpeg)
    # Decode via ffmpeg → raw; avoid Pillow dependency in CI.
    raw = tmp_path / "preview.raw"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(preview),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(raw),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        pytest.skip("could not decode preview for emoji check")
    data = raw.read_bytes()
    # Count near-white pixels in the top band (where caption sits).
    width, height = 1080, 1920
    white = 0
    band_end = int(height * 0.22) * width * 3
    for i in range(0, min(band_end, len(data)), 3):
        if data[i] > 200 and data[i + 1] > 200 and data[i + 2] > 200:
            white += 1
    # Plain "Wait  for it " letters alone are well under this; emoji silhouettes push higher.
    assert white > 3500, f"expected emoji silhouettes, white_pixels={white}"


@pytest.mark.asyncio
async def test_burn_text_keeps_vertical_resolution(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    _make_vertical_mp4(src, seconds=2.0)
    data = src.read_bytes()
    style = TextStyle(font_size=56, padding_top=10, outline_width=3)
    prepared = prepare_segments(
        text=None,
        segments=[
            TextSegment(start_seconds=0, end_seconds=1, text="WAIT FOR IT"),
            TextSegment(start_seconds=1, end_seconds=2, text="Подожди секунду"),
        ],
        style=style,
        frame_width=1080,
        duration_seconds=2.0,
    )
    rendered = await burn_text_video(
        data,
        style=style,
        segments=prepared,
        width=1080,
        height=1920,
        output_format="mp4",
        preview_at_seconds=0.2,
        source_suffix=".mp4",
    )
    probed = await probe_bytes(rendered.video_bytes, suffix=".mp4")
    assert probed.width == 1080
    assert probed.height == 1920
    assert probed.duration_seconds == pytest.approx(2.0, abs=0.35)
    assert rendered.preview_jpeg is not None
    assert rendered.preview_jpeg[:2] == b"\xff\xd8"


@pytest.mark.asyncio
async def test_export_without_text_does_not_mutate_source(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    _make_vertical_mp4(src, seconds=1.0)
    original = src.read_bytes()
    rendered = await export_without_text(original, output_format="mp4")
    assert original == src.read_bytes()
    assert rendered.video_bytes
    assert rendered.extension == ".mp4"
    probed = await probe_bytes(rendered.video_bytes, suffix=".mp4")
    assert probed.width == 1080
    assert probed.height == 1920


@pytest.mark.asyncio
async def test_service_skip_text_and_mp4_format(
    settings: Settings, storage: LocalFileStorage, tmp_path: Path
) -> None:
    src = tmp_path / "clip.mp4"
    _make_vertical_mp4(src, seconds=1.5)
    key = f"hook_video/jobs/{uuid4()}/output.mp4"
    await storage.save_file(key, src, content_type="video/mp4")
    url = storage.get_url(key)

    service = TextRendererService(storage=storage, settings=settings)
    from app.stages.text_renderer.schemas import TextRenderRequest

    response = await service.render(
        TextRenderRequest(
            video_id=uuid4(),
            generated_video_url=url,
            skip_text=True,
            output_format="mp4",
        )
    )
    assert response.status == "success"
    assert response.format == "mp4"
    assert response.text_segments_rendered == 0
    assert response.width == 1080
    assert response.height == 1920
    # Source archive exists and original media key is untouched.
    assert await storage.exists(key)
    media = storage.get_local_path(key).read_bytes()
    assert media == src.read_bytes()
