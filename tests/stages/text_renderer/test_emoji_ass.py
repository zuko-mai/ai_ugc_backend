from app.stages.text_renderer.ass import PreparedSegment, build_ass_document
from app.stages.text_renderer.emoji_ass import contains_emoji, wrap_emoji_for_ass
from app.stages.text_renderer.fonts import resolve_emoji_font_path
from app.stages.text_renderer.schemas import TextStyle


def test_wrap_emoji_tags_runs_and_escapes_plain_text() -> None:
    out = wrap_emoji_for_ass("Wait 🔥 for {it} 😍")
    assert r"{\fnNoto Emoji}🔥{\r}" in out
    assert r"{\fnNoto Emoji}😍{\r}" in out
    assert r"\{it\}" in out
    assert "Wait " in out


def test_contains_emoji() -> None:
    assert contains_emoji("nope") is False
    assert contains_emoji("hey ✨") is True
    assert contains_emoji("👩‍💻") is True


def test_ass_document_uses_emoji_font_override() -> None:
    style = TextStyle(font_size=48)
    doc = build_ass_document(
        width=1080,
        height=1920,
        style=style,
        font_name="Arial",
        emoji_font_name="Noto Emoji",
        segments=[
            PreparedSegment(
                start_seconds=0,
                end_seconds=2,
                original_text="Wait 🔥",
                lines=["Wait 🔥"],
            )
        ],
    )
    assert r"{\fnNoto Emoji}🔥{\r}" in doc
    assert "Style: Default,Arial," in doc


def test_bundled_emoji_font_resolves() -> None:
    path = resolve_emoji_font_path()
    assert path is not None
    assert path.is_file()
