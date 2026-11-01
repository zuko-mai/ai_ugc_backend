import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


def _make_clip(
    tmp_path: Path,
    *,
    name: str,
    color: str,
    size: str,
    seconds: float,
) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    out = tmp_path / name
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d={seconds}:r=30",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=520:duration={seconds}",
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
        pytest.skip(f"failed to generate clip: {result.stderr[:300]}")
    return out


def _store_hook(
    settings: Settings,
    tmp_path: Path,
    *,
    name: str,
    color: str,
    size: str,
    seconds: float,
) -> str:
    out = _make_clip(tmp_path, name=name, color=color, size=size, seconds=seconds)
    key = f"test_inputs/{uuid4()}/{name}"
    dest = settings.storage_root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out.read_bytes())
    return f"{settings.public_base_url}/media/{key}"


def _compose_multipart(
    *,
    project_id: str,
    hook_url: str,
    demo_path: Path,
    settings: dict | None = None,
) -> dict:
    settings = settings or {}
    data = {
        "project_id": project_id,
        "hook_asset_id": str(uuid4()),
        "hook_video_url": hook_url,
        "output_format": settings.get("output_format", "mp4"),
        "aspect_ratio": settings.get("aspect_ratio", "9:16"),
        "resolution": settings.get("resolution", "1080x1920"),
        "fps": str(settings.get("fps", 30)),
        "transition": settings.get("transition", "none"),
        "transition_duration_seconds": str(
            settings.get("transition_duration_seconds", 0)
        ),
        "audio_mode": settings.get("audio_mode", "both"),
    }
    files = {
        "demo_video": (demo_path.name, demo_path.read_bytes(), "video/mp4"),
    }
    return {"data": data, "files": files}


def test_compose_hard_cut_async_job(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    hook_url = _store_hook(
        settings, tmp_path, name="hook.mp4", color="red", size="540x960", seconds=1.0
    )
    demo_path = _make_clip(
        tmp_path, name="demo.mp4", color="blue", size="540x960", seconds=2.0
    )
    project_id = str(uuid4())
    form = _compose_multipart(
        project_id=project_id,
        hook_url=hook_url,
        demo_path=demo_path,
        settings={
            "resolution": "540x960",
            "aspect_ratio": "9:16",
            "fps": 30,
            "transition": "none",
            "transition_duration_seconds": 0,
            "audio_mode": "both",
        },
    )
    accepted = client.post(
        "/v1/product-demo-composer/compose",
        data=form["data"],
        files=form["files"],
    )
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    assert body["project_id"] == project_id
    assert body["status"] == "queued"
    job_id = body["job_id"]

    job = client.get(f"/v1/product-demo-composer/jobs/{job_id}")
    assert job.status_code == 200, job.text
    result = job.json()
    assert result["status"] == "success"
    assert result["project_id"] == project_id
    assert result["asset_id"]
    assert result["video_url"].startswith("http://testserver/media/product_demo_composer/")
    assert result["width"] == 540
    assert result["height"] == 960
    assert result["duration_seconds"] == pytest.approx(3.0, abs=0.35)
    assert len(result["segments"]) == 2
    assert result["segments"][0]["type"] == "hook"
    assert result["segments"][1]["type"] == "product_demo"
    assert result["segments"][0]["end_seconds"] == pytest.approx(1.0, abs=0.15)
    assert result["segments"][1]["start_seconds"] == pytest.approx(1.0, abs=0.15)

    media = client.get(result["video_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/")

    assert client.get(hook_url.removeprefix("http://testserver")).status_code == 200

    stored = client.get(f"/v1/product-demo-composer/jobs/{job_id}/request")
    assert stored.status_code == 200
    assert stored.json()["demo_video_url"].startswith(
        "http://testserver/media/product_demo_composer/inputs/"
    )


def test_rerun_with_fade_settings(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    hook_url = _store_hook(
        settings, tmp_path, name="hook2.mp4", color="#112233", size="320x568", seconds=1.0
    )
    demo_path = _make_clip(
        tmp_path, name="demo2.mp4", color="#445566", size="320x568", seconds=1.0
    )
    form = _compose_multipart(
        project_id=str(uuid4()),
        hook_url=hook_url,
        demo_path=demo_path,
        settings={
            "resolution": "320x568",
            "aspect_ratio": "9:16",
            "transition": "none",
            "transition_duration_seconds": 0,
        },
    )
    accepted = client.post(
        "/v1/product-demo-composer/compose",
        data=form["data"],
        files=form["files"],
    )
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]
    first = client.get(f"/v1/product-demo-composer/jobs/{job_id}")
    assert first.status_code == 200
    assert first.json()["status"] == "success"

    rerun = client.post(
        f"/v1/product-demo-composer/jobs/{job_id}/rerun",
        json={
            "settings": {
                "resolution": "320x568",
                "aspect_ratio": "9:16",
                "transition": "fade",
                "transition_duration_seconds": 0.2,
                "audio_mode": "demo",
            }
        },
    )
    assert rerun.status_code == 202, rerun.text
    new_job_id = rerun.json()["job_id"]
    assert new_job_id != job_id

    second = client.get(f"/v1/product-demo-composer/jobs/{new_job_id}")
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["status"] == "success"
    assert body["duration_seconds"] == pytest.approx(2.0, abs=0.35)
    assert "rerun_from_archived_inputs" in body["warnings"]

    stored = client.get(f"/v1/product-demo-composer/jobs/{new_job_id}/request")
    assert stored.status_code == 200
    assert stored.json()["settings"]["transition"] == "fade"
    assert stored.json()["settings"]["audio_mode"] == "demo"


def test_missing_hook_returns_404(
    client: TestClient, tmp_path: Path
) -> None:
    demo_path = _make_clip(
        tmp_path, name="demo.mp4", color="blue", size="320x568", seconds=0.5
    )
    form = _compose_multipart(
        project_id=str(uuid4()),
        hook_url="http://testserver/media/missing/hook.mp4",
        demo_path=demo_path,
        settings={"resolution": "320x568", "aspect_ratio": "9:16"},
    )
    response = client.post(
        "/v1/product-demo-composer/compose",
        data=form["data"],
        files=form["files"],
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "video_fetch_failed"


def test_job_not_found(client: TestClient) -> None:
    response = client.get(f"/v1/product-demo-composer/jobs/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "job_not_found"


def test_invalid_transition_settings_rejected(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    hook_url = _store_hook(
        settings, tmp_path, name="hook.mp4", color="red", size="320x568", seconds=0.5
    )
    demo_path = _make_clip(
        tmp_path, name="demo.mp4", color="blue", size="320x568", seconds=0.5
    )
    form = _compose_multipart(
        project_id=str(uuid4()),
        hook_url=hook_url,
        demo_path=demo_path,
        settings={
            "resolution": "320x568",
            "aspect_ratio": "9:16",
            "transition": "fade",
            "transition_duration_seconds": 0,
        },
    )
    response = client.post(
        "/v1/product-demo-composer/compose",
        data=form["data"],
        files=form["files"],
    )
    assert response.status_code == 422


def test_unsupported_demo_extension_rejected(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    hook_url = _store_hook(
        settings, tmp_path, name="hook.mp4", color="red", size="320x568", seconds=0.5
    )
    bad = tmp_path / "demo.txt"
    bad.write_text("not a video")
    data = {
        "project_id": str(uuid4()),
        "hook_asset_id": str(uuid4()),
        "hook_video_url": hook_url,
        "resolution": "320x568",
        "aspect_ratio": "9:16",
        "transition": "none",
        "transition_duration_seconds": "0",
    }
    response = client.post(
        "/v1/product-demo-composer/compose",
        data=data,
        files={"demo_video": ("demo.txt", bad.read_bytes(), "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "unsupported_format"
