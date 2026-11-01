import base64
import json

import httpx
import pytest

from app.core.config import Settings
from app.stages.hook_video_generator.exceptions import (
    ProviderApiError,
    ProviderNotConfiguredError,
)
from app.stages.hook_video_generator.instructions import MotionInstructions
from app.stages.hook_video_generator.providers.base import ProviderImage, VideoProviderRequest
from app.stages.hook_video_generator.providers.kling import (
    KlingVideoProvider,
    build_kling_prompt,
    map_kling_duration,
    map_kling_resolution,
    resolve_kling_model,
)
from app.stages.hook_video_generator.providers.registry import get_provider, list_providers


def _request(**overrides: object) -> VideoProviderRequest:
    payload = {
        "first_frame": ProviderImage(
            data=b"frame-bytes",
            mime_type="image/jpeg",
            width=1080,
            height=1920,
            source_url="http://localhost:8000/media/frame.jpg",
        ),
        "instructions": MotionInstructions(
            prompt="Subtle nod. Keep identity. Locked camera.",
            negative_prompt="identity drift, zoom",
            identity_lock="lock",
            camera_lock="camera",
            timeline=(),
            warnings=(),
        ),
        "duration_seconds": 3,
        "aspect_ratio": "9:16",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "model_name": "kling-3.0",
        "timeout_seconds": 5,
    }
    payload.update(overrides)
    return VideoProviderRequest(**payload)  # type: ignore[arg-type]


def test_model_and_resolution_helpers() -> None:
    assert resolve_kling_model("kling-3.0")[1] == "/image-to-video/kling-3.0"
    assert resolve_kling_model("kling-3.0-turbo")[1] == "/image-to-video/kling-3.0-turbo"
    with pytest.raises(ProviderApiError):
        resolve_kling_model("kling-4.0")
    assert map_kling_resolution(1080, 1920) == "1080p"
    assert map_kling_resolution(720, 1280) == "720p"
    assert map_kling_duration(3) == (3, [])
    assert map_kling_duration(3.4)[0] == 3
    with pytest.raises(ProviderApiError):
        map_kling_duration(2)


def test_prompt_appends_negative_and_truncates() -> None:
    request = _request()
    prompt = build_kling_prompt(request)
    assert "Subtle nod" in prompt
    assert "Avoid: identity drift, zoom." in prompt

    huge = "x" * 4000
    long_request = _request(
        instructions=MotionInstructions(
            prompt=huge,
            negative_prompt="zoom",
            identity_lock="lock",
            camera_lock="camera",
            timeline=(),
            warnings=(),
        )
    )
    assert len(build_kling_prompt(long_request)) <= 3072


def test_registry_requires_api_key(settings: Settings) -> None:
    assert "kling" in list_providers()
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("kling", settings)
    settings.kling_api_key = "test-key"
    provider = get_provider("kling", settings)
    assert isinstance(provider, KlingVideoProvider)


async def test_submit_builds_kling_payload_and_stays_queued() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/image-to-video/kling-3.0"
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content.decode())
        seen["body"] = body
        return httpx.Response(
            200,
            json={
                "code": 0,
                "message": "SUCCEED",
                "data": {"id": "task-77", "status": "submitted"},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(
            api_key="secret",
            base_url="https://kling.test",
            client=client,
        )
        submission = await provider.submit(_request())

    assert submission.provider_job_id == "task-77"
    assert submission.status == "queued"
    body = seen["body"]
    assert body["settings"] == {
        "resolution": "1080p",
        "duration": 3,
        "audio": "off",
        "multi_shot": False,
    }
    assert body["contents"][0]["type"] == "prompt"
    assert "Avoid: identity drift, zoom." in body["contents"][0]["text"]
    assert body["contents"][1]["type"] == "first_frame"
    frame = body["contents"][1]["url"]
    assert frame.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(frame.split(",", 1)[1]) == b"frame-bytes"
    assert "video_url" not in body


async def test_submit_uses_public_https_frame_url() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"code": 0, "data": {"id": "task-88", "status": "processing"}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(api_key="secret", base_url="https://kling.test", client=client)
        submission = await provider.submit(
            _request(
                first_frame=ProviderImage(
                    data=b"frame-bytes",
                    mime_type="image/jpeg",
                    width=1080,
                    height=1920,
                    source_url="https://cdn.example.com/frame.jpg",
                )
            )
        )
    assert submission.status == "processing"
    assert seen["body"]["contents"][1]["url"] == "https://cdn.example.com/frame.jpg"


async def test_submit_rejects_duration_two() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(api_key="secret", base_url="https://kling.test", client=client)
        with pytest.raises(ProviderApiError, match="3 or 4"):
            await provider.submit(_request(duration_seconds=2))


async def test_poll_returns_video_url_only_when_succeeded() -> None:
    states = iter(
        [
            {
                "code": 0,
                "data": [{"id": "task-77", "status": "processing"}],
            },
            {
                "code": 0,
                "data": [
                    {
                        "id": "task-77",
                        "status": "succeeded",
                        "outputs": [
                            {
                                "type": "video",
                                "url": "https://cdn.kling.example/out.mp4",
                                "duration": "3",
                            }
                        ],
                    }
                ],
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tasks"
        assert request.url.params["task_ids"] == "task-77"
        return httpx.Response(200, json=next(states))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(api_key="secret", base_url="https://kling.test", client=client)
        processing = await provider.poll("task-77")
        ready = await provider.poll("task-77")

    assert processing.status == "processing"
    assert processing.video_url is None
    assert ready.status == "succeeded"
    assert ready.video_url == "https://cdn.kling.example/out.mp4"
    assert ready.video_bytes is None


async def test_poll_failed_and_api_error() -> None:
    def failed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": [
                    {
                        "id": "task-1",
                        "status": "failed",
                        "message": "risk control",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(failed)
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(api_key="secret", base_url="https://kling.test", client=client)
        result = await provider.poll("task-1")
    assert result.status == "failed"
    assert result.error_message == "risk control"

    def api_error(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 1002, "message": "bad token"})

    transport = httpx.MockTransport(api_error)
    async with httpx.AsyncClient(transport=transport, base_url="https://kling.test") as client:
        provider = KlingVideoProvider(api_key="secret", base_url="https://kling.test", client=client)
        with pytest.raises(ProviderApiError, match="bad token"):
            await provider.poll("task-1")
