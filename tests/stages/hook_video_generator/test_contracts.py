from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.stages.hook_video_generator.schemas import (
    HookVideoAccepted,
    HookVideoGenerateRequest,
    HookVideoJobView,
)


def test_generate_request_defaults_match_hook_output_spec() -> None:
    request = HookVideoGenerateRequest(
        video_id=uuid4(),
        first_frame_url="http://localhost:8000/media/frame.jpg",
    )
    assert request.duration_seconds == 3
    assert request.aspect_ratio == "9:16"
    assert request.output_resolution == "1080x1920"
    assert request.fps == 30
    assert request.model_provider == "kling"
    assert request.model_name == "kling-3.0"
    assert request.motion_plan == {}


def test_duration_is_limited_to_hook_length() -> None:
    video_id = uuid4()
    url = "http://localhost/frame.jpg"
    for duration in (2, 3, 4):
        request = HookVideoGenerateRequest(
            video_id=video_id,
            first_frame_url=url,
            duration_seconds=duration,
        )
        assert request.duration_seconds == duration
    for duration in (1, 1.9, 4.1, 8):
        with pytest.raises(ValidationError):
            HookVideoGenerateRequest(
                video_id=video_id,
                first_frame_url=url,
                duration_seconds=duration,
            )


def test_resolution_must_match_aspect_and_be_even() -> None:
    video_id = uuid4()
    url = "http://localhost/frame.jpg"
    with pytest.raises(ValidationError):
        HookVideoGenerateRequest(
            video_id=video_id,
            first_frame_url=url,
            aspect_ratio="9:16",
            output_resolution="1920x1080",
        )
    with pytest.raises(ValidationError):
        HookVideoGenerateRequest(
            video_id=video_id,
            first_frame_url=url,
            aspect_ratio="9:16",
            output_resolution="361x640",
        )
    landscape = HookVideoGenerateRequest(
        video_id=video_id,
        first_frame_url=url,
        aspect_ratio="16:9",
        output_resolution="1920x1080",
    )
    assert landscape.output_resolution == "1920x1080"


def test_accepted_response_cannot_claim_success() -> None:
    with pytest.raises(ValidationError):
        HookVideoAccepted(
            job_id=uuid4(),
            video_id=uuid4(),
            status="success",
            model_provider="stub",
            model_name="stub-hold",
        )


def test_success_view_requires_a_validated_file() -> None:
    with pytest.raises(ValidationError):
        HookVideoJobView(
            job_id=uuid4(),
            video_id=uuid4(),
            status="success",
            model_provider="stub",
            model_name="stub-hold",
        )
    with pytest.raises(ValidationError):
        HookVideoJobView(
            job_id=uuid4(),
            video_id=uuid4(),
            status="processing",
            generated_video_url="http://localhost/video.mp4",
            model_provider="stub",
            model_name="stub-hold",
        )
    ready = HookVideoJobView(
        job_id=uuid4(),
        video_id=uuid4(),
        status="success",
        generated_video_url="http://localhost/video.mp4",
        duration_seconds=3,
        width=1080,
        height=1920,
        fps=30,
        model_provider="stub",
        model_name="stub-hold",
    )
    assert ready.status == "success"
    assert ready.width == 1080
    assert ready.height == 1920
