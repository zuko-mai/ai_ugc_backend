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
        face_replacement_provider="stub",
        face_replacement_model="stub-passthrough",
        face_replacement_timeout_seconds=5,
        max_face_image_size_mb=5,
        min_selected_frame_width=64,
        min_selected_frame_height=64,
        min_face_reference_width=64,
        min_face_reference_height=64,
        hook_video_provider="stub",
        hook_video_model="stub-hold",
        hook_video_timeout_seconds=30,
        hook_video_min_duration_seconds=2,
        hook_video_max_duration_seconds=4,
        hook_video_duration_tolerance_seconds=0.25,
        hook_video_http_submit_url="",
        hook_video_http_status_url="",
        hook_video_http_api_key="",
        max_hook_frame_size_mb=5,
        max_text_render_video_size_mb=20,
        max_product_demo_video_size_mb=50,
        max_music_overlay_video_size_mb=50,
        max_music_overlay_audio_size_mb=20,
        kling_api_key="",
        kling_api_base_url="https://api-singapore.klingai.com",
        kling_audio="off",
        kling_multi_shot=False,
        kling_prefer_base64=False,
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
    monkeypatch.setattr("app.stages.hook_analyzer.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.hook_text_engine.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.face_replacement.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.hook_video_generator.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.text_renderer.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.product_demo_composer.deps.get_settings", _settings)
    monkeypatch.setattr("app.stages.music_overlay.deps.get_settings", _settings)
    monkeypatch.setattr("app.pipeline.deps.get_settings", _settings)

    from app.api import deps

    deps.get_storage.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    deps.get_storage.cache_clear()
    get_settings.cache_clear()
