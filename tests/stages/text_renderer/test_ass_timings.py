from app.stages.text_renderer.ass import (
    _ass_time,
    build_ass_document,
    prepare_segments,
)
from app.stages.text_renderer.schemas import TextSegment, TextStyle


def test_ass_time_formatting() -> None:
    assert _ass_time(0) == "0:00:00.00"
    assert _ass_time(2) == "0:00:02.00"
    assert _ass_time(1.5) == "0:00:01.50"
    assert _ass_time(65.03) == "0:01:05.03"


def test_prepare_segments_clips_to_duration() -> None:
    style = TextStyle(font_size=40)
    prepared = prepare_segments(
        text=None,
        segments=[
            TextSegment(start_seconds=0, end_seconds=2, text="first"),
            TextSegment(start_seconds=2.5, end_seconds=9, text="second"),
            TextSegment(start_seconds=10, end_seconds=12, text="out of range"),
        ],
        style=style,
        frame_width=1080,
        duration_seconds=3.0,
    )
    assert len(prepared) == 2
    assert prepared[0].end_seconds == 2
    assert prepared[1].end_seconds == 3.0
    assert prepared[0].original_text == "first"
    assert prepared[1].original_text == "second"


def test_fallback_text_covers_full_clip() -> None:
    style = TextStyle()
    prepared = prepare_segments(
        text="WAIT FOR IT",
        segments=[],
        style=style,
        frame_width=1080,
        duration_seconds=3.2,
    )
    assert len(prepared) == 1
    assert prepared[0].start_seconds == 0
    assert prepared[0].end_seconds == 3.2
    assert prepared[0].original_text == "WAIT FOR IT"


def test_null_end_seconds_holds_until_video_end() -> None:
    style = TextStyle()
    prepared = prepare_segments(
        text=None,
        segments=[
            TextSegment(start_seconds=0, end_seconds=None, text="full clip caption"),
        ],
        style=style,
        frame_width=1080,
        duration_seconds=4.7,
    )
    assert len(prepared) == 1
    assert prepared[0].start_seconds == 0
    assert prepared[0].end_seconds == 4.7


def test_ass_style_is_bold_by_default() -> None:
    style = TextStyle()
    prepared = prepare_segments(
        text="POV caption",
        segments=[],
        style=style,
        frame_width=1080,
        duration_seconds=2,
    )
    doc = build_ass_document(
        width=1080,
        height=1920,
        style=style,
        font_name="Arial",
        segments=prepared,
    )
    # Bold flag is the 8th style field value (-1).
    assert ",-1,0,0,0,100,100," in doc
    assert style.font_weight == "bold"
    assert style.font_size == 72
    assert style.padding_top == 72


def test_padding_top_maps_to_ass_margin_v() -> None:
    style = TextStyle(padding_top=25, font_size=40)
    prepared = prepare_segments(
        text="hook",
        segments=[],
        style=style,
        frame_width=1080,
        duration_seconds=2,
    )
    doc = build_ass_document(
        width=1080,
        height=1920,
        style=style,
        font_name="Arial",
        segments=prepared,
    )
    # 25% of 1920 = 480 → MarginV in the style line.
    assert ",8," in doc  # center-top alignment
    assert ",480,1" in doc or doc.count("480") >= 1


def test_ass_document_contains_escaped_unicode_and_timings() -> None:
    style = TextStyle(
        font_family="Arial",
        font_size=48,
        font_color="#FFFFFF",
        outline_color="#000000",
        outline_width=2,
        padding_top=8,
    )
    prepared = prepare_segments(
        text=None,
        segments=[
            TextSegment(start_seconds=0, end_seconds=1.25, text="Привет {world}"),
            TextSegment(start_seconds=1.25, end_seconds=3, text="line1\nline2"),
        ],
        style=style,
        frame_width=1080,
        duration_seconds=3,
    )
    doc = build_ass_document(
        width=1080,
        height=1920,
        style=style,
        font_name="Arial",
        segments=prepared,
    )
    assert "PlayResX: 1080" in doc
    assert "PlayResY: 1920" in doc
    assert "0:00:00.00,0:00:01.25" in doc
    assert "0:00:01.25,0:00:03.00" in doc
    assert "Привет" in doc
    assert r"\{world\}" in doc
    assert r"line1\Nline2" in doc
    # Wording is preserved (escaped braces only).
    assert "{world}" not in doc.replace(r"\{world\}", "")
