from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.stages.music_overlay.schemas import OverlayRequest, OverlaySettings


def test_example_payload_keeps_offsets_independent() -> None:
    payload = OverlayRequest(
        project_id=uuid4(),
        video={"asset_id": uuid4(), "video_url": "https://cdn.example/clip.mp4"},
        music={"asset_id": uuid4(), "audio_url": "https://cdn.example/song.mp3"},
        settings={
            "music_start_offset_seconds": 35,
            "video_start_seconds": 0,
            "music_volume": 0.3,
            "original_audio_enabled": True,
            "original_audio_volume": 1.0,
            "fade_in_seconds": 0,
            "fade_out_seconds": 1.5,
            "loop_music": False,
            "music_end_behavior": "silence",
            "output_format": "mp4",
        },
    )
    assert payload.settings.music_start_offset_seconds == 35
    assert payload.settings.video_start_seconds == 0
    assert payload.settings.music_volume == 0.3
    assert payload.settings.music_end_behavior == "silence"
    assert payload.settings.loop_music is False


def test_loop_flag_selects_loop_behavior() -> None:
    settings = OverlaySettings(loop_music=True, music_end_behavior="silence")
    assert settings.music_end_behavior == "loop"
    assert settings.loop_music is True


def test_explicit_loop_behavior_sets_flag() -> None:
    settings = OverlaySettings(loop_music=False, music_end_behavior="loop")
    assert settings.loop_music is True


def test_loop_conflicts_with_trim() -> None:
    with pytest.raises(ValidationError):
        OverlaySettings(loop_music=True, music_end_behavior="trim")


def test_volume_and_offset_bounds() -> None:
    with pytest.raises(ValidationError):
        OverlaySettings(music_volume=1.5)
    with pytest.raises(ValidationError):
        OverlaySettings(music_start_offset_seconds=-1)
    with pytest.raises(ValidationError):
        OverlaySettings(video_start_seconds=-0.1)


def test_trim_keeps_loop_disabled() -> None:
    settings = OverlaySettings(music_end_behavior="trim")
    assert settings.loop_music is False
    assert settings.music_end_behavior == "trim"


def test_urls_are_stripped() -> None:
    payload = OverlayRequest(
        project_id=uuid4(),
        video={"asset_id": uuid4(), "video_url": "  https://cdn.example/a.mp4  "},
        music={"asset_id": uuid4(), "audio_url": " https://cdn.example/a.wav "},
    )
    assert payload.video.video_url == "https://cdn.example/a.mp4"
    assert payload.music.audio_url == "https://cdn.example/a.wav"
