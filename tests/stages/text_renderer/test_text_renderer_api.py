import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


def _store_vertical_video(settings: Settings, tmp_path: Path, *, seconds: float = 3.0) -> str:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    out = tmp_path / "generated.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=#224466:s=540x960:d={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"failed to generate test video: {result.stderr[:300]}")

    key = f"test_inputs/{uuid4()}/generated.mp4"
    dest = settings.storage_root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out.read_bytes())
    return f"{settings.public_base_url}/media/{key}"


def test_render_with_segments_and_preview(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    url = _store_vertical_video(settings, tmp_path, seconds=3.0)
    video_id = str(uuid4())
    payload = {
        "video_id": video_id,
        "generated_video_url": url,
        "text": "fallback unused",
        "language": "ru",
        "style": {
            "font_family": "Arial",
            "font_size": 42,
            "font_color": "#FFFFFF",
            "outline_color": "#000000",
            "outline_width": 2,
            "padding_top": 12,
            "text_align": "center",
        },
        "segments": [
            {"start_seconds": 0, "end_seconds": 1.2, "text": "Подожди…"},
            {"start_seconds": 1.2, "end_seconds": 3.0, "text": "никто не говорит об этом"},
        ],
        "output_format": "mp4",
    }
    response = client.post("/v1/text-renderer/render", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == video_id
    assert body["status"] == "success"
    assert body["format"] == "mp4"
    assert body["text_segments_rendered"] == 2
    assert body["width"] == 540
    assert body["height"] == 960
    assert body["duration_seconds"] == pytest.approx(3.0, abs=0.4)
    assert body["rendered_video_url"].startswith("http://testserver/media/text_renderer/")
    assert body["preview_image_url"]
    assert body["render_id"]

    media = client.get(body["rendered_video_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/")

    preview = client.get(body["preview_image_url"].removeprefix("http://testserver"))
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/")


def test_preview_endpoint(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    url = _store_vertical_video(settings, tmp_path, seconds=2.0)
    payload = {
        "video_id": str(uuid4()),
        "generated_video_url": url,
        "text": "WAIT FOR IT",
        "language": "en",
        "style": {"padding_top": 78, "font_size": 36},
        "preview_at_seconds": 0.5,
        "output_format": "mp4",
    }
    response = client.post("/v1/text-renderer/preview", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["text_segments_rendered"] == 1
    assert body["preview_at_seconds"] == 0.5
    assert body["preview_image_url"].startswith("http://testserver/media/text_renderer/")


def test_skip_text_export(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    url = _store_vertical_video(settings, tmp_path, seconds=1.0)
    payload = {
        "video_id": str(uuid4()),
        "generated_video_url": url,
        "skip_text": True,
        "output_format": "mp4",
    }
    response = client.post("/v1/text-renderer/render", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text_segments_rendered"] == 0
    assert body["status"] == "success"


def test_rerender_without_regenerating_source(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    url = _store_vertical_video(settings, tmp_path, seconds=2.0)
    video_id = str(uuid4())
    first = client.post(
        "/v1/text-renderer/render",
        json={
            "video_id": video_id,
            "generated_video_url": url,
            "text": "first caption",
            "style": {"padding_top": 10, "font_size": 40},
            "output_format": "mp4",
        },
    )
    assert first.status_code == 200, first.text
    render_id = first.json()["render_id"]

    second = client.post(
        f"/v1/text-renderer/renders/{render_id}/rerender",
        json={
            "text": "second caption exact",
            "style": {
                "font_family": "Arial",
                "font_size": 52,
                "font_color": "#FFFF00",
                "outline_color": "#000000",
                "outline_width": 3,
                "padding_top": 74,
            },
        },
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["video_id"] == video_id
    assert body["render_id"] != render_id
    assert body["text_segments_rendered"] == 1

    stored = client.get(f"/v1/text-renderer/renders/{body['render_id']}/request")
    assert stored.status_code == 200
    assert stored.json()["text"] == "second caption exact"
    assert stored.json()["style"]["padding_top"] == 74


def test_render_from_previous_output_reuses_clean_source(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """Passing a burned output URL must not stack a second caption on top."""
    url = _store_vertical_video(settings, tmp_path, seconds=2.0)
    video_id = str(uuid4())
    first = client.post(
        "/v1/text-renderer/render",
        json={
            "video_id": video_id,
            "generated_video_url": url,
            "text": "OLD TOP TEXT",
            "style": {"padding_top": 8, "font_size": 40},
            "output_format": "mp4",
        },
    )
    assert first.status_code == 200, first.text
    burned_url = first.json()["rendered_video_url"]
    first_source = client.get(
        f"/v1/text-renderer/renders/{first.json()['render_id']}/request"
    ).json()["source_video_key"]

    second = client.post(
        "/v1/text-renderer/render",
        json={
            "video_id": video_id,
            "generated_video_url": burned_url,
            "text": "NEW BOTTOM TEXT",
            "style": {"padding_top": 74, "font_size": 64, "font_weight": "bold"},
            "output_format": "mp4",
        },
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert any("reused_clean_source_from_previous_render" in w for w in body["warnings"])
    second_source = client.get(
        f"/v1/text-renderer/renders/{body['render_id']}/request"
    ).json()["source_video_key"]
    assert second_source == first_source


def test_validation_rejects_empty_text_without_skip(client: TestClient) -> None:
    response = client.post(
        "/v1/text-renderer/render",
        json={
            "video_id": str(uuid4()),
            "generated_video_url": "http://testserver/media/missing.mp4",
            "output_format": "mp4",
        },
    )
    assert response.status_code == 422


def test_validation_rejects_bad_color(client: TestClient) -> None:
    response = client.post(
        "/v1/text-renderer/render",
        json={
            "video_id": str(uuid4()),
            "generated_video_url": "http://testserver/media/x.mp4",
            "text": "hi",
            "style": {"font_color": "white"},
            "output_format": "mp4",
        },
    )
    assert response.status_code == 422


def test_get_render_not_found(client: TestClient) -> None:
    response = client.get(f"/v1/text-renderer/renders/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["error_code"] == "render_not_found"
