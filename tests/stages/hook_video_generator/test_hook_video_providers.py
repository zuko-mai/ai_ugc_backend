import base64
import json
import shutil
from dataclasses import fields

import httpx
import pytest

from app.core.config import Settings
from app.stages.hook_video_generator.exceptions import ProviderApiError, ProviderNotConfiguredError
from app.stages.hook_video_generator.instructions import MotionInstructions
from app.stages.hook_video_generator.providers.base import (
    ProviderImage,
    VideoProviderRequest,
    VideoProviderSubmission,
    normalize_remote_status,
)
from app.stages.hook_video_generator.providers.http import HttpVideoProvider
from app.stages.hook_video_generator.providers.registry import get_provider
from app.stages.hook_video_generator.providers.stub import StubVideoProvider
from app.stages.hook_video_generator.probe import probe_bytes


def _request() -> VideoProviderRequest:
    return VideoProviderRequest(
        first_frame=ProviderImage(
            data=b"\xff\xd8\xff\xd9",
            mime_type="image/jpeg",
            width=360,
            height=640,
            source_url="http://example/frame.jpg",
        ),
        instructions=MotionInstructions(
            prompt="Hold identity.",
            negative_prompt="identity drift",
            identity_lock="lock",
            camera_lock="camera",
            timeline=(),
            warnings=(),
        ),
        duration_seconds=2,
        aspect_ratio="9:16",
        width=360,
        height=640,
        fps=30,
        model_name="unit-model",
        timeout_seconds=5,
    )


def test_submission_has_no_video_payload() -> None:
    names = {item.name for item in fields(VideoProviderSubmission)}
    assert "video_bytes" not in names
    assert "video_url" not in names
    assert "generated_video_url" not in names


async def test_stub_submit_is_only_a_queued_task() -> None:
    provider = StubVideoProvider(polls_until_ready=2)
    submission = await provider.submit(_request())
    assert submission.status == "queued"
    assert submission.provider_job_id
    update = await provider.poll(submission.provider_job_id)
    assert update.status == "processing"
    assert update.video_bytes is None
    assert update.video_url is None


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)
async def test_stub_second_poll_returns_a_real_video() -> None:
    import cv2
    import numpy as np

    image = np.full((640, 360, 3), 40, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    request = _request()
    request = VideoProviderRequest(
        first_frame=ProviderImage(
            data=encoded.tobytes(),
            mime_type="image/jpeg",
            width=360,
            height=640,
            source_url="http://example/frame.jpg",
        ),
        instructions=request.instructions,
        duration_seconds=2,
        aspect_ratio="9:16",
        width=360,
        height=640,
        fps=30,
        model_name="unit-model",
        timeout_seconds=5,
    )
    provider = StubVideoProvider(polls_until_ready=2)
    submission = await provider.submit(request)
    await provider.poll(submission.provider_job_id)
    ready = await provider.poll(submission.provider_job_id)
    assert ready.status == "succeeded"
    assert ready.video_bytes
    probed = await probe_bytes(ready.video_bytes)
    assert abs(probed.duration_seconds - 2) <= 0.25
    assert probed.width == 360
    assert probed.height == 640


def test_normalize_remote_status() -> None:
    assert normalize_remote_status("queued") == "queued"
    assert normalize_remote_status("completed") == "succeeded"
    assert normalize_remote_status("SUCCESS") == "succeeded"
    assert normalize_remote_status("in progress") == "processing"
    with pytest.raises(ValueError):
        normalize_remote_status("mystery")


def test_unknown_and_unconfigured_providers(settings: Settings) -> None:
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("missing-vendor", settings)
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("http", settings)


def _instructions_request(image: bytes = b"frame-bytes") -> VideoProviderRequest:
    base = _request()
    return VideoProviderRequest(
        first_frame=ProviderImage(
            data=image,
            mime_type="image/jpeg",
            width=360,
            height=640,
            source_url="http://example/frame.jpg",
        ),
        instructions=base.instructions,
        duration_seconds=base.duration_seconds,
        aspect_ratio=base.aspect_ratio,
        width=base.width,
        height=base.height,
        fps=base.fps,
        model_name="vendor-model-a",
        timeout_seconds=5,
    )


async def test_http_submit_queues_and_ignores_a_video_url() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content.decode())
            seen["body"] = body
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={"id": "task-1", "status": "queued", "video_url": "https://cdn.example/early.mp4"},
            )
        return httpx.Response(200, json={"status": "processing"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = HttpVideoProvider(
            submit_url="https://video.example/submit",
            status_url_template="https://video.example/tasks/{provider_job_id}",
            api_key="secret",
            timeout_seconds=5,
            client=client,
        )
        submission = await provider.submit(_instructions_request())
        assert submission.status == "queued"
        assert submission.provider_job_id == "task-1"
        update = await provider.poll(submission.provider_job_id)
        assert update.status == "processing"
        assert update.video_url is None

    body = seen["body"]
    assert body["model"] == "vendor-model-a"
    assert body["prompt"] == "Hold identity."
    assert body["negative_prompt"] == "identity drift"
    assert body["duration_seconds"] == 2
    assert body["image_mime_type"] == "image/jpeg"
    assert base64.b64decode(body["image_base64"]) == b"frame-bytes"
    assert "video_bytes" not in body
    assert "video_url" not in body
    assert seen["auth"] == "Bearer secret"


async def test_http_submit_rejects_an_immediate_success() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "task-2", "status": "succeeded"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = HttpVideoProvider(
            submit_url="https://video.example/submit",
            status_url_template="https://video.example/tasks/{provider_job_id}",
            api_key="",
            timeout_seconds=5,
            client=client,
        )
        with pytest.raises(ProviderApiError):
            await provider.submit(_instructions_request())


async def test_http_poll_returns_the_finished_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-3", "status": "processing"})
        assert request.url.path.endswith("/task-3")
        return httpx.Response(
            200,
            json={"status": "completed", "output_url": "https://cdn.example/done.mp4"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = HttpVideoProvider(
            submit_url="https://video.example/submit",
            status_url_template="https://video.example/tasks/{provider_job_id}",
            api_key="",
            timeout_seconds=5,
            client=client,
        )
        submission = await provider.submit(_instructions_request(b"x"))
        update = await provider.poll(submission.provider_job_id)
    assert submission.status == "processing"
    assert update.status == "succeeded"
    assert update.video_url == "https://cdn.example/done.mp4"
    assert update.video_bytes is None


def test_status_template_must_include_the_job_placeholder() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        HttpVideoProvider(
            submit_url="https://video.example/submit",
            status_url_template="https://video.example/tasks",
            api_key="",
            timeout_seconds=5,
        )
