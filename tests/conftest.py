import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.storage.local import LocalFileStorage


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        log_level="WARNING",
        storage_root=tmp_path / "storage",
        public_base_url="http://testserver",
        max_upload_size_mb=10,
        download_timeout_seconds=10,
    )


@pytest.fixture
def storage(settings: Settings) -> LocalFileStorage:
    return LocalFileStorage(
        root=settings.storage_root,
        public_base_url=settings.public_base_url,
    )


@pytest.fixture
def sample_mp4(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required to generate sample media")

    out = tmp_path / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"failed to generate sample mp4: {result.stderr[:300]}")
    return out


@pytest.fixture
def sample_mp4_no_audio(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required to generate sample media")

    out = tmp_path / "silent.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x120:d=0.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"failed to generate silent mp4: {result.stderr[:300]}")
    return out


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    get_settings.cache_clear()

    def _settings() -> Settings:
        return settings

    monkeypatch.setattr("app.core.config.get_settings", _settings)
    monkeypatch.setattr("app.api.deps.get_settings", _settings)
    monkeypatch.setattr("app.main.get_settings", _settings)
    monkeypatch.setattr("app.stages.video_ingestion.deps.get_settings", _settings)

    from app.api import deps

    deps.get_storage.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    deps.get_storage.cache_clear()
    get_settings.cache_clear()
