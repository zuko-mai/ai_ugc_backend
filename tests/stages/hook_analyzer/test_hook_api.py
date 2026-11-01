from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient


def _ingest_sample(client: TestClient, sample_mp4: Path) -> tuple[UUID, str]:
    with sample_mp4.open("rb") as fh:
        response = client.post(
            "/v1/ingestion/upload",
            files={"file": ("sample.mp4", fh, "video/mp4")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    return UUID(body["video_id"]), body["video_url"]


@pytest.fixture
def mock_gemini(monkeypatch: pytest.MonkeyPatch):
    from app.stages.hook_analyzer.gemini_motion import GeminiMotionResult
    from app.stages.hook_analyzer.gemini_pricing import estimate_cost

    async def _fake_prompt(**kwargs):  # noqa: ANN003
        assert kwargs["video_path"].is_file()
        return GeminiMotionResult(
            motion_prompt=(
                "At the start the subject looks toward camera with a soft neutral expression, "
                "then the mouth begins to move as if speaking while the head stays mostly still."
            ),
            usage=estimate_cost(
                model=kwargs.get("model", "gemini-3.8-flash"),
                prompt_tokens=1200,
                candidates_tokens=180,
                thoughts_tokens=40,
            ),
        )

    monkeypatch.setattr(
        "app.stages.hook_analyzer.service.generate_kling_motion_prompt",
        _fake_prompt,
    )


def test_candidates_motion_plan_and_reanalyze(
    client: TestClient, sample_mp4: Path, mock_gemini
) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)

    candidates_resp = client.post(
        "/v1/hook-analyzer/candidates",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "hook_start_seconds": 0,
            "hook_end_seconds": 1,
            "candidate_count": 3,
        },
    )
    assert candidates_resp.status_code == 200, candidates_resp.text
    candidates_body = candidates_resp.json()
    assert candidates_body["video_id"] == str(video_id)
    assert 1 <= len(candidates_body["candidates"]) <= 3
    assert candidates_body["analysis_key"]

    first = candidates_body["candidates"][0]
    assert "frame_id" in first
    assert "timestamp_seconds" in first
    assert first["image_url"].startswith("http://testserver/media/hooks/")
    assert 0 <= first["quality_score"] <= 1
    assert first["reason"]

    media = client.get(first["image_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")

    motion_resp = client.post(
        "/v1/hook-analyzer/motion-plan",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "start_seconds": 0,
            "duration_seconds": 1,
        },
    )
    assert motion_resp.status_code == 200, motion_resp.text
    motion = motion_resp.json()
    assert motion["video_id"] == str(video_id)
    assert motion["interval"]["start_seconds"] == 0
    assert motion["interval"]["end_seconds"] == 1
    assert motion["motionPrompt"]
    assert "camera" in motion["motionPrompt"].lower() or "subject" in motion["motionPrompt"].lower()
    assert motion["trimmed_video_url"].startswith("http://testserver/media/hooks/")
    assert motion["gemini_usage"]["total_tokens"] > 0
    assert motion["gemini_usage"]["estimated_cost_usd"] >= 0
    assert motion["gemini_usage"]["currency"] == "USD"
    assert motion["motion_timeline"] == []

    trimmed = client.get(motion["trimmed_video_url"].removeprefix("http://testserver"))
    assert trimmed.status_code == 200
    assert trimmed.headers["content-type"].startswith("video/")

    reanalyze_resp = client.post(
        "/v1/hook-analyzer/reanalyze",
        json={
            "video_id": str(video_id),
            "selected_frame_id": first["frame_id"],
            "start_seconds": 0,
            "duration_seconds": 1,
        },
    )
    assert reanalyze_resp.status_code == 200, reanalyze_resp.text
    reanalyzed = reanalyze_resp.json()
    assert reanalyzed["selected_frame_id"] == first["frame_id"]
    assert reanalyzed["motionPrompt"]
    assert reanalyzed["motion_timeline"] == []


def test_reanalyze_without_candidates_returns_404(client: TestClient) -> None:
    response = client.post(
        "/v1/hook-analyzer/reanalyze",
        json={
            "video_id": str(uuid4()),
            "selected_frame_id": str(uuid4()),
            "start_seconds": 0,
            "duration_seconds": 3,
        },
    )
    assert response.status_code == 404
    assert response.json()["error_code"] in {"session_not_found", "frame_not_found"}


def test_invalid_interval_rejected(client: TestClient, sample_mp4: Path) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)
    response = client.post(
        "/v1/hook-analyzer/candidates",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
            "hook_start_seconds": 3,
            "hook_end_seconds": 1,
            "candidate_count": 5,
        },
    )
    assert response.status_code == 422


def test_motion_plan_defaults_start_and_duration(
    client: TestClient, sample_mp4: Path, mock_gemini
) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)
    # Sample is 1s long; request default 3s duration — trim should still succeed
    # (ffmpeg will output available frames).
    response = client.post(
        "/v1/hook-analyzer/motion-plan",
        json={
            "video_id": str(video_id),
            "video_url": video_url,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interval"]["start_seconds"] == 0
    assert body["interval"]["end_seconds"] == 3
    assert body["motionPrompt"]
    assert body["trimmed_video_url"]


def test_motion_plan_by_video_id_only(
    client: TestClient, sample_mp4: Path, mock_gemini
) -> None:
    video_id, _video_url = _ingest_sample(client, sample_mp4)
    response = client.post(
        "/v1/hook-analyzer/motion-plan",
        json={
            "video_id": str(video_id),
            "start_seconds": 0,
            "duration_seconds": 1,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["video_id"] == str(video_id)
    assert response.json()["motionPrompt"]


def test_motion_plan_by_video_url_only(
    client: TestClient, sample_mp4: Path, mock_gemini
) -> None:
    video_id, video_url = _ingest_sample(client, sample_mp4)
    response = client.post(
        "/v1/hook-analyzer/motion-plan",
        json={
            "video_url": video_url,
            "start_seconds": 0,
            "duration_seconds": 1,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["video_id"] == str(video_id)


def test_motion_plan_requires_id_or_url(client: TestClient) -> None:
    response = client.post(
        "/v1/hook-analyzer/motion-plan",
        json={"start_seconds": 0, "duration_seconds": 1},
    )
    assert response.status_code == 422
