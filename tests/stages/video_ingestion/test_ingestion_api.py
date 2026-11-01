from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_success(client: TestClient, sample_mp4: Path) -> None:
    with sample_mp4.open("rb") as fh:
        response = client.post(
            "/v1/ingestion/upload",
            files={"file": ("sample.mp4", fh, "video/mp4")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["video_id"]
    assert body["video_url"].startswith("http://testserver/media/originals/")
    assert body["width"] == 320
    assert body["height"] == 240
    assert body["has_audio"] is True
    assert body["audio_url"] is None
    assert body["duration_seconds"] > 0
    assert body["fps"] > 0


def test_upload_rejects_txt(client: TestClient, tmp_path: Path) -> None:
    bad = tmp_path / "note.txt"
    bad.write_text("not a video")
    with bad.open("rb") as fh:
        response = client.post(
            "/v1/ingestion/upload",
            files={"file": ("note.txt", fh, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_format"


def test_upload_rejects_fake_mp4(client: TestClient, tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not an mp4 container at all!!!!!")
    with fake.open("rb") as fh:
        response = client.post(
            "/v1/ingestion/upload",
            files={"file": ("fake.mp4", fh, "video/mp4")},
        )
    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_format"


def test_ingest_from_url(
    client: TestClient,
    sample_mp4: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator

    from app.core.config import Settings

    async def fake_stream_from_url(url: str, settings: Settings):
        async def _iter() -> AsyncIterator[bytes]:
            data = sample_mp4.read_bytes()
            for i in range(0, len(data), 64 * 1024):
                yield data[i : i + 64 * 1024]

        return ".mp4", "video/mp4", _iter()

    monkeypatch.setattr(
        "app.stages.video_ingestion.service.stream_from_url",
        fake_stream_from_url,
    )

    response = client.post(
        "/v1/ingestion/url",
        json={"url": "https://cdn.example.com/videos/clip.mp4"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["has_audio"] is True


def test_media_serve(client: TestClient, sample_mp4: Path) -> None:
    with sample_mp4.open("rb") as fh:
        ingest = client.post(
            "/v1/ingestion/upload",
            files={"file": ("sample.mp4", fh, "video/mp4")},
        )
    video_url = ingest.json()["video_url"]
    path = video_url.removeprefix("http://testserver")
    media = client.get(path)
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/")
