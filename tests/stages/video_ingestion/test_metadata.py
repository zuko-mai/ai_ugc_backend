import shutil
from pathlib import Path

import pytest

from app.stages.video_ingestion.metadata import extract_metadata


@pytest.mark.asyncio
async def test_extract_metadata_with_audio(sample_mp4: Path) -> None:
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe required")

    meta = await extract_metadata(sample_mp4)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.duration_seconds > 0
    assert meta.fps > 0
    assert meta.has_audio is True
    assert meta.video_codec is not None
    assert meta.audio_codec is not None


@pytest.mark.asyncio
async def test_extract_metadata_without_audio(sample_mp4_no_audio: Path) -> None:
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe required")

    meta = await extract_metadata(sample_mp4_no_audio)
    assert meta.has_audio is False
    assert meta.audio_codec is None
