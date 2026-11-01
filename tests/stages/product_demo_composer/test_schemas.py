from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.stages.product_demo_composer.schemas import ComposeRequest, ComposeSettings


def test_default_settings_are_hard_cut() -> None:
    settings = ComposeSettings()
    assert settings.transition == "none"
    assert settings.transition_duration_seconds == 0
    assert settings.audio_mode == "both"
    assert settings.resolution == "1080x1920"
    assert settings.fps == 30


def test_fade_requires_positive_duration() -> None:
    with pytest.raises(ValidationError):
        ComposeSettings(transition="fade", transition_duration_seconds=0)


def test_none_rejects_nonzero_duration() -> None:
    with pytest.raises(ValidationError):
        ComposeSettings(transition="none", transition_duration_seconds=0.5)


def test_resolution_must_match_aspect() -> None:
    with pytest.raises(ValidationError):
        ComposeSettings(aspect_ratio="9:16", resolution="1920x1080")


def test_compose_request_accepts_valid_payload() -> None:
    payload = ComposeRequest(
        project_id=uuid4(),
        hook_video={"asset_id": uuid4(), "video_url": "https://example.com/hook.mp4"},
        demo_video={"asset_id": uuid4(), "video_url": "https://example.com/demo.mp4"},
        settings={
            "output_format": "mp4",
            "aspect_ratio": "9:16",
            "resolution": "1080x1920",
            "fps": 30,
            "transition": "crossfade",
            "transition_duration_seconds": 0.4,
            "audio_mode": "hook",
        },
    )
    assert payload.settings.transition == "crossfade"
    assert payload.settings.audio_mode == "hook"
