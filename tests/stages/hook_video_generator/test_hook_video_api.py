import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.stages.hook_video_generator.motion_prompt import CAMERA_LOCK, IDENTITY_LOCK
from app.stages.hook_video_generator.providers.base import (
    VideoGenerationProvider,
    VideoProviderPoll,
    VideoProviderRequest,
    VideoProviderSubmission,
)
from app.stages.hook_video_generator.providers.registry import register_provider, unregister_provider
from app.stages.hook_video_generator.providers.stub import STUB_HOLD_WARNING, STUB_QUEUED_WARNING
from app.stages.hook_video_generator.render import render_still_video

_NEEDS_FFMPEG = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)


@pytest.fixture
def frame_url(settings: Settings, tmp_path: Path) -> str:
    image = np.full((640, 360, 3), 70, dtype=np.uint8)
    cv2.circle(image, (180, 220), 60, (180, 160, 140), -1)
    path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(path), image)
    key = "test_inputs/hook_frame.jpg"
    dest = settings.storage_root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
    return f"{settings.public_base_url}/media/{key}"


def _payload(video_id: str, frame_url: str, **overrides: object) -> dict:
    body = {
        "video_id": video_id,
        "first_frame_url": frame_url,
        "motion_plan": {
            "emotion": "curious",
            "emotion_intensity": 0.6,
            "gaze_direction": "toward camera",
            "facial_expression": "slight smile",
            "motion_timeline": [
                {
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "head_motion": "slight nod",
                    "eye_motion": "looks at camera",
                    "mouth_motion": "closed",
                    "body_motion": "still",
                    "emotion": "curious",
                }
            ],
        },
        "duration_seconds": 2,
        "aspect_ratio": "9:16",
        "output_resolution": "360x640",
        "fps": 30,
        "model_provider": "stub",
        "model_name": "stub-hold",
    }
    body.update(overrides)
    return body


def _frame_bytes(settings: Settings) -> bytes:
    return (settings.storage_root / "test_inputs" / "hook_frame.jpg").read_bytes()


def test_submit_stores_inputs_and_is_not_a_finished_video(
    client: TestClient,
    settings: Settings,
    frame_url: str,
) -> None:
    video_id = str(uuid4())
    response = client.post("/v1/hook-video/generate", json=_payload(video_id, frame_url))
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert accepted["video_id"] == video_id
    assert accepted["model_provider"] == "stub"
    assert "generated_video_url" not in accepted
    assert STUB_QUEUED_WARNING in accepted["warnings"]

    stored = client.get(f"/v1/hook-video/jobs/{accepted['job_id']}/request")
    assert stored.status_code == 200, stored.text
    request_body = stored.json()
    assert request_body["motion_plan"]["emotion"] == "curious"
    assert "slight nod" in request_body["compiled_prompt"]
    assert IDENTITY_LOCK in request_body["compiled_prompt"]
    assert CAMERA_LOCK in request_body["compiled_prompt"]
    assert (settings.storage_root / request_body["first_frame_key"]).is_file()

    processing = client.get(f"/v1/hook-video/jobs/{accepted['job_id']}")
    assert processing.status_code == 200, processing.text
    assert processing.json()["status"] == "processing"
    assert processing.json()["generated_video_url"] is None


@_NEEDS_FFMPEG
def test_poll_returns_success_only_after_the_file_is_validated(
    client: TestClient,
    frame_url: str,
) -> None:
    video_id = str(uuid4())
    accepted = client.post("/v1/hook-video/generate", json=_payload(video_id, frame_url))
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job_id"]

    client.get(f"/v1/hook-video/jobs/{job_id}")
    ready = client.get(f"/v1/hook-video/jobs/{job_id}")
    assert ready.status_code == 200, ready.text
    body = ready.json()
    assert body["status"] == "success"
    assert body["video_id"] == video_id
    assert body["model_provider"] == "stub"
    assert body["model_name"] == "stub-hold"
    assert body["width"] == 360
    assert body["height"] == 640
    assert abs(body["duration_seconds"] - 2) <= 0.25
    assert abs(body["fps"] - 30) <= 1
    assert body["generated_video_url"].startswith("http://testserver/media/hook_video/")
    assert body["error_message"] is None
    assert STUB_HOLD_WARNING in body["warnings"]

    media = client.get(body["generated_video_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/")
    assert len(media.content) > 100

    again = client.get(f"/v1/hook-video/jobs/{job_id}")
    assert again.json() == body


@_NEEDS_FFMPEG
def test_rerun_uses_the_stored_frame_and_can_change_model(
    client: TestClient,
    frame_url: str,
) -> None:
    video_id = str(uuid4())
    original = client.post("/v1/hook-video/generate", json=_payload(video_id, frame_url))
    assert original.status_code == 202, original.text
    original_id = original.json()["job_id"]
    original_request = client.get(f"/v1/hook-video/jobs/{original_id}/request").json()

    rerun = client.post(
        f"/v1/hook-video/jobs/{original_id}/rerun",
        json={"model_name": "stub-hold-rerun"},
    )
    assert rerun.status_code == 202, rerun.text
    rerun_body = rerun.json()
    assert rerun_body["job_id"] != original_id
    assert rerun_body["status"] == "queued"
    assert rerun_body["model_name"] == "stub-hold-rerun"
    assert "generated_video_url" not in rerun_body

    rerun_request = client.get(f"/v1/hook-video/jobs/{rerun_body['job_id']}/request").json()
    assert rerun_request["first_frame_key"] == original_request["first_frame_key"]
    assert rerun_request["motion_plan"] == original_request["motion_plan"]

    job_id = rerun_body["job_id"]
    client.get(f"/v1/hook-video/jobs/{job_id}")
    ready = client.get(f"/v1/hook-video/jobs/{job_id}").json()
    assert ready["status"] == "success"
    assert ready["model_name"] == "stub-hold-rerun"
    assert ready["generated_video_url"]


def test_compare_queues_each_model_without_a_video(
    client: TestClient,
    frame_url: str,
) -> None:
    video_id = str(uuid4())
    payload = _payload(video_id, frame_url)
    payload.pop("model_provider")
    payload.pop("model_name")
    payload["models"] = [
        {"model_provider": "stub", "model_name": "model-a"},
        {"model_provider": "stub", "model_name": "model-b"},
    ]
    response = client.post("/v1/hook-video/compare", json=payload)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["video_id"] == video_id
    assert len(body["jobs"]) == 2
    assert {item["status"] for item in body["jobs"]} == {"queued"}
    assert {item["model_name"] for item in body["jobs"]} == {"model-a", "model-b"}
    assert "generated_video_url" not in body["jobs"][0]

    prompts = [
        client.get(f"/v1/hook-video/jobs/{item['job_id']}/request").json()["compiled_prompt"]
        for item in body["jobs"]
    ]
    assert prompts[0] == prompts[1]
    assert IDENTITY_LOCK in prompts[0]


@_NEEDS_FFMPEG
def test_compare_poll_completes_each_model(client: TestClient, frame_url: str) -> None:
    video_id = str(uuid4())
    payload = _payload(video_id, frame_url)
    payload.pop("model_provider")
    payload.pop("model_name")
    payload["models"] = [
        {"model_provider": "stub", "model_name": "model-a"},
        {"model_provider": "stub", "model_name": "model-b"},
    ]
    accepted = client.post("/v1/hook-video/compare", json=payload).json()
    comparison_id = accepted["comparison_id"]
    client.get(f"/v1/hook-video/comparisons/{comparison_id}")
    ready = client.get(f"/v1/hook-video/comparisons/{comparison_id}")
    assert ready.status_code == 200, ready.text
    jobs = ready.json()["jobs"]
    assert {item["status"] for item in jobs} == {"success"}
    assert {item["model_name"] for item in jobs} == {"model-a", "model-b"}
    assert len({item["generated_video_url"] for item in jobs}) == 2


def test_invalid_duration_aspect_and_provider_are_rejected(
    client: TestClient,
    frame_url: str,
) -> None:
    video_id = str(uuid4())
    too_short = client.post(
        "/v1/hook-video/generate",
        json=_payload(video_id, frame_url, duration_seconds=1),
    )
    assert too_short.status_code == 422

    mismatch = client.post(
        "/v1/hook-video/generate",
        json=_payload(video_id, frame_url, output_resolution="1920x1080"),
    )
    assert mismatch.status_code == 422

    unknown = client.post(
        "/v1/hook-video/generate",
        json=_payload(video_id, frame_url, model_provider="missing-vendor"),
    )
    assert unknown.status_code == 503
    assert unknown.json()["error_code"] == "provider_not_configured"


def test_missing_frame_and_unknown_job(client: TestClient, settings: Settings) -> None:
    missing = client.post(
        "/v1/hook-video/generate",
        json=_payload(str(uuid4()), f"{settings.public_base_url}/media/missing/frame.jpg"),
    )
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "frame_fetch_failed"

    unknown = client.get(f"/v1/hook-video/jobs/{uuid4()}")
    assert unknown.status_code == 404
    assert unknown.json()["error_code"] == "job_not_found"

    providers = client.get("/v1/hook-video/providers")
    assert providers.status_code == 200
    assert "stub" in providers.json()["providers"]
    assert "http" in providers.json()["providers"]
    assert "kling" in providers.json()["providers"]


def test_callback_failure_does_not_publish_a_video(
    client: TestClient,
    frame_url: str,
) -> None:
    accepted = client.post(
        "/v1/hook-video/generate",
        json=_payload(str(uuid4()), frame_url),
    ).json()
    response = client.post(
        f"/v1/hook-video/jobs/{accepted['job_id']}/callback",
        json={"provider_status": "failed", "error": "vendor rejected"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["generated_video_url"] is None
    assert body["error_message"] == "vendor rejected"

    follow = client.get(f"/v1/hook-video/jobs/{accepted['job_id']}")
    assert follow.json()["status"] == "failed"


def test_unrecognized_callback_status_is_rejected(client: TestClient, frame_url: str) -> None:
    accepted = client.post(
        "/v1/hook-video/generate",
        json=_payload(str(uuid4()), frame_url),
    ).json()
    response = client.post(
        f"/v1/hook-video/jobs/{accepted['job_id']}/callback",
        json={"provider_status": "not-a-status"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "hook_video_error"


@_NEEDS_FFMPEG
async def test_callback_success_still_validates_the_file(
    client: TestClient,
    settings: Settings,
    frame_url: str,
) -> None:
    short = await render_still_video(
        _frame_bytes(settings),
        width=360,
        height=640,
        fps=30,
        duration_seconds=1,
    )
    short_key = "hook_video_test/short.mp4"
    short_path = settings.storage_root / short_key
    short_path.parent.mkdir(parents=True, exist_ok=True)
    short_path.write_bytes(short)
    short_url = f"{settings.public_base_url}/media/{short_key}"

    accepted = client.post(
        "/v1/hook-video/generate",
        json=_payload(str(uuid4()), frame_url),
    ).json()
    rejected = client.post(
        f"/v1/hook-video/jobs/{accepted['job_id']}/callback",
        json={"provider_status": "succeeded", "video_url": short_url},
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "failed"
    assert rejected.json()["generated_video_url"] is None
    assert "duration" in rejected.json()["error_message"]

    good = await render_still_video(
        _frame_bytes(settings),
        width=360,
        height=640,
        fps=30,
        duration_seconds=2,
    )
    good_key = "hook_video_test/good.mp4"
    (settings.storage_root / good_key).write_bytes(good)
    good_url = f"{settings.public_base_url}/media/{good_key}"
    fresh = client.post(
        "/v1/hook-video/generate",
        json=_payload(str(uuid4()), frame_url),
    ).json()
    ready = client.post(
        f"/v1/hook-video/jobs/{fresh['job_id']}/callback",
        json={"provider_status": "success", "video_url": good_url},
    )
    assert ready.status_code == 200, ready.text
    body = ready.json()
    assert body["status"] == "success"
    assert body["generated_video_url"]
    assert abs(body["duration_seconds"] - 2) <= 0.25
    assert "completion_via_callback" in body["warnings"]
    assert STUB_HOLD_WARNING not in body["warnings"]


class _TerminalSubmitProvider(VideoGenerationProvider):
    name = "terminal-submit"

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        del request
        return VideoProviderSubmission(provider_job_id="early", status="succeeded")  # type: ignore[arg-type]

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        raise AssertionError(f"poll should not run for {provider_job_id}")


class _EmptyVideoProvider(VideoGenerationProvider):
    name = "empty-video"

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        del request
        return VideoProviderSubmission(provider_job_id="empty-1", status="queued")

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        return VideoProviderPoll(provider_job_id=provider_job_id, status="succeeded")


class _ShortVideoProvider(VideoGenerationProvider):
    name = "short-video"
    _request: VideoProviderRequest | None = None

    async def submit(self, request: VideoProviderRequest) -> VideoProviderSubmission:
        self._request = request
        return VideoProviderSubmission(provider_job_id="short-1", status="queued")

    async def poll(self, provider_job_id: str) -> VideoProviderPoll:
        assert self._request is not None
        data = await render_still_video(
            self._request.first_frame.data,
            width=self._request.width,
            height=self._request.height,
            fps=self._request.fps,
            duration_seconds=1,
        )
        return VideoProviderPoll(
            provider_job_id=provider_job_id,
            status="succeeded",
            video_bytes=data,
            mime_type="video/mp4",
        )


def test_submit_terminal_status_is_not_success(client: TestClient, frame_url: str) -> None:
    provider = _TerminalSubmitProvider()
    register_provider("terminal-submit", lambda _settings: provider)
    try:
        response = client.post(
            "/v1/hook-video/generate",
            json=_payload(str(uuid4()), frame_url, model_provider="terminal-submit"),
        )
        assert response.status_code == 502, response.text
        assert response.json()["error_code"] == "generation_failed"
        job_id = response.json()["details"]["job_id"]
        follow = client.get(f"/v1/hook-video/jobs/{job_id}")
        assert follow.json()["status"] == "failed"
        assert follow.json()["generated_video_url"] is None
    finally:
        unregister_provider("terminal-submit")


def test_poll_without_a_file_fails_the_job(client: TestClient, frame_url: str) -> None:
    provider = _EmptyVideoProvider()
    register_provider("empty-video", lambda _settings: provider)
    try:
        accepted = client.post(
            "/v1/hook-video/generate",
            json=_payload(str(uuid4()), frame_url, model_provider="empty-video"),
        )
        assert accepted.status_code == 202, accepted.text
        ready = client.get(f"/v1/hook-video/jobs/{accepted.json()['job_id']}")
        assert ready.json()["status"] == "failed"
        assert ready.json()["generated_video_url"] is None
        assert "without a video payload" in ready.json()["error_message"]
    finally:
        unregister_provider("empty-video")


@_NEEDS_FFMPEG
def test_poll_rejects_a_file_with_the_wrong_duration(
    client: TestClient,
    frame_url: str,
) -> None:
    provider = _ShortVideoProvider()
    register_provider("short-video", lambda _settings: provider)
    try:
        accepted = client.post(
            "/v1/hook-video/generate",
            json=_payload(
                str(uuid4()),
                frame_url,
                model_provider="short-video",
                model_name="short-model",
            ),
        )
        assert accepted.status_code == 202, accepted.text
        ready = client.get(f"/v1/hook-video/jobs/{accepted.json()['job_id']}")
        assert ready.status_code == 200, ready.text
        body = ready.json()
        assert body["status"] == "failed"
        assert body["generated_video_url"] is None
        assert "duration_mismatch" in body["error_message"]
    finally:
        unregister_provider("short-video")
