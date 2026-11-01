import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.stages.music_overlay.exceptions import OverlayError
from tests.stages.music_overlay.test_overlay import _silence_then_tone, _video, _wav


def _store(settings: Settings, source: Path, name: str) -> str:
    key = f"test_inputs/{uuid4()}/{name}"
    dest = settings.storage_root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())
    return f"{settings.public_base_url}/media/{key}"


def _payload(project_id: str, video_url: str, audio_url: str, settings: dict | None = None) -> dict:
    body = {
        "project_id": project_id,
        "video": {"asset_id": str(uuid4()), "video_url": video_url},
        "music": {"asset_id": str(uuid4()), "audio_url": audio_url},
        "settings": {
            "music_start_offset_seconds": 0,
            "video_start_seconds": 0,
            "music_volume": 0.3,
            "original_audio_enabled": True,
            "original_audio_volume": 1.0,
            "fade_in_seconds": 0,
            "fade_out_seconds": 0.2,
            "loop_music": False,
            "music_end_behavior": "silence",
            "output_format": "mp4",
        },
    }
    if settings:
        body["settings"].update(settings)
    return body


def test_overlay_json_job_returns_asset_and_keeps_sources(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.2, with_audio=True, size="160x120")
    music = _wav(tmp_path / "music.wav", seconds=3.0)
    video_url = _store(settings, video, "source.mp4")
    audio_url = _store(settings, music, "song.wav")
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    music_hash = hashlib.sha256(music.read_bytes()).hexdigest()
    project_id = str(uuid4())

    accepted = client.post(
        "/v1/music-overlay/overlay",
        json=_payload(
            project_id,
            video_url,
            audio_url,
            {
                "music_start_offset_seconds": 0.4,
                "video_start_seconds": 0.2,
                "music_volume": 0.3,
                "fade_out_seconds": 0.2,
            },
        ),
    )
    assert accepted.status_code == 202, accepted.text
    queued = accepted.json()
    assert queued["project_id"] == project_id
    assert queued["status"] == "queued"
    assert "music_trimmed_to_video" in queued["warnings"]

    job_id = queued["job_id"]
    job = client.get(f"/v1/music-overlay/jobs/{job_id}")
    assert job.status_code == 200, job.text
    result = job.json()
    assert result["status"] == "success"
    assert result["project_id"] == project_id
    assert result["job_id"] == job_id
    assert result["asset_id"]
    assert result["video_url"].startswith("http://testserver/media/music_overlay/")
    assert result["duration_seconds"] == pytest.approx(1.2, abs=0.25)
    assert result["audio"] == {
        "music_start_offset_seconds": 0.4,
        "video_start_seconds": 0.2,
        "music_volume": 0.3,
        "original_audio_enabled": True,
        "fade_in_seconds": 0,
        "fade_out_seconds": 0.2,
    }

    media = client.get(result["video_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/")

    stored_video = client.get(video_url.removeprefix("http://testserver"))
    stored_music = client.get(audio_url.removeprefix("http://testserver"))
    assert hashlib.sha256(stored_video.content).hexdigest() == video_hash
    assert hashlib.sha256(stored_music.content).hexdigest() == music_hash

    archived = client.get(f"/v1/music-overlay/jobs/{job_id}/request")
    assert archived.status_code == 200
    assert archived.json()["video_key"].startswith("music_overlay/inputs/")
    assert archived.json()["settings"]["music_start_offset_seconds"] == 0.4


def test_upload_and_rerun(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.0, with_audio=False)
    music = _wav(tmp_path / "upload.wav", seconds=1.5)
    video_url = _store(settings, video, "clip.mp4")
    project_id = str(uuid4())

    accepted = client.post(
        "/v1/music-overlay/overlay/upload",
        data={
            "project_id": project_id,
            "video_asset_id": str(uuid4()),
            "video_url": video_url,
            "music_volume": "0.4",
            "original_audio_enabled": "false",
            "music_end_behavior": "silence",
            "fade_in_seconds": "0",
            "fade_out_seconds": "0",
        },
        files={"music": ("upload.wav", music.read_bytes(), "audio/wav")},
    )
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]
    first = client.get(f"/v1/music-overlay/jobs/{job_id}")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "success"
    assert first.json()["audio"]["original_audio_enabled"] is False
    assert first.json()["audio"]["music_volume"] == 0.4

    rerun = client.post(
        f"/v1/music-overlay/jobs/{job_id}/rerun",
        json={
            "settings": {
                "music_volume": 0.8,
                "original_audio_enabled": False,
                "music_end_behavior": "loop",
                "output_format": "mp4",
            }
        },
    )
    assert rerun.status_code == 202, rerun.text
    new_id = rerun.json()["job_id"]
    assert new_id != job_id
    second = client.get(f"/v1/music-overlay/jobs/{new_id}")
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["status"] == "success"
    assert body["audio"]["music_volume"] == 0.8
    assert "rerun_from_archived_inputs" in body["warnings"]


def test_offset_beyond_music_is_rejected(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    video = _video(tmp_path / "video.mp4", seconds=1.0, with_audio=False)
    music = _silence_then_tone(tmp_path / "music.wav")
    response = client.post(
        "/v1/music-overlay/overlay",
        json=_payload(
            str(uuid4()),
            _store(settings, video, "v.mp4"),
            _store(settings, music, "m.wav"),
            {"music_start_offset_seconds": 30, "fade_out_seconds": 0},
        ),
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error_code"] == "invalid_overlay_settings"
    assert body["status"] == "error"


def test_unknown_job_and_bad_upload(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    missing = client.get(f"/v1/music-overlay/jobs/{uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "job_not_found"

    video = _video(tmp_path / "video.mp4", seconds=0.5, with_audio=False)
    rejected = client.post(
        "/v1/music-overlay/overlay/upload",
        data={
            "project_id": str(uuid4()),
            "video_asset_id": str(uuid4()),
            "video_url": _store(settings, video, "v.mp4"),
        },
        files={"music": ("notes.txt", b"not audio", "text/plain")},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == "unsupported_format"

    gone = client.post(
        "/v1/music-overlay/overlay",
        json=_payload(
            str(uuid4()),
            f"{settings.public_base_url}/media/missing/{uuid4()}/nope.mp4",
            _store(settings, _wav(tmp_path / "m.wav", seconds=0.5), "m.wav"),
            {"fade_out_seconds": 0},
        ),
    )
    assert gone.status_code == 404
    assert gone.json()["error_code"] == "media_fetch_failed"


def test_ffmpeg_failure_marks_job_failed(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(**_kwargs):
        raise OverlayError("ffmpeg failed while mixing music onto the video")

    monkeypatch.setattr("app.stages.music_overlay.service.overlay_music", _boom)
    video = _video(tmp_path / "video.mp4", seconds=0.6, with_audio=False)
    music = _wav(tmp_path / "music.wav", seconds=0.6)
    accepted = client.post(
        "/v1/music-overlay/overlay",
        json=_payload(
            str(uuid4()),
            _store(settings, video, "v.mp4"),
            _store(settings, music, "m.wav"),
            {"fade_out_seconds": 0, "music_volume": 0.2},
        ),
    )
    assert accepted.status_code == 202, accepted.text
    job = client.get(f"/v1/music-overlay/jobs/{accepted.json()['job_id']}")
    assert job.status_code == 200, job.text
    body = job.json()
    assert body["status"] == "failed"
    assert body["video_url"] is None
    assert "ffmpeg failed" in body["error_message"]
