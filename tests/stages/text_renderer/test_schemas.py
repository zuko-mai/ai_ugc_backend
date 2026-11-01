from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.stages.text_renderer.schemas import TextRenderRequest, TextSegment, TextStyle


def test_segment_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        TextSegment(start_seconds=2, end_seconds=1, text="x")


def test_style_rejects_path_like_font() -> None:
    with pytest.raises(ValidationError):
        TextStyle(font_family="../etc/passwd")


def test_request_allows_skip_without_text() -> None:
    req = TextRenderRequest(
        video_id=uuid4(),
        generated_video_url="http://localhost:8000/media/a.mp4",
        skip_text=True,
    )
    assert req.skip_text is True


def test_text_control_chars_rejected() -> None:
    with pytest.raises(ValidationError):
        TextSegment(start_seconds=0, end_seconds=1, text="bad\x01text")


def test_exact_text_preserved_including_unicode() -> None:
    seg = TextSegment(
        start_seconds=0,
        end_seconds=1.5,
        text="Wait… никто не говорит\nоб этом?",
    )
    assert seg.text == "Wait… никто не говорит\nоб этом?"


def test_segment_allows_null_end_seconds() -> None:
    seg = TextSegment(start_seconds=0, end_seconds=None, text="until end")
    assert seg.end_seconds is None


def test_default_style_matches_ugc_reference() -> None:
    style = TextStyle()
    assert style.font_size >= 64
    assert style.font_weight in {"bold", "black"}
    assert 60 <= style.padding_top <= 85
    assert style.outline_width >= 3


def test_padding_top_percent_bounds() -> None:
    assert TextStyle(padding_top=0).padding_top == 0
    assert TextStyle(padding_top=95).padding_top == 95
    with pytest.raises(ValidationError):
        TextStyle(padding_top=96)
    with pytest.raises(ValidationError):
        TextStyle(padding_top=-1)
