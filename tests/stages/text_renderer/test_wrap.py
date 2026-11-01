from app.stages.text_renderer.wrap import wrap_text


def test_wrap_preserves_words_and_punctuation() -> None:
    text = "Wait… nobody talks about this AquaBottle?"
    lines = wrap_text(text, font_size=48, frame_width=320, max_width_ratio=0.86)
    # Soft-wrap / hard-split may insert breaks, but every original character remains.
    assert "".join(lines).replace(" ", "") == text.replace(" ", "").replace("\n", "")
    assert "AquaBottle" in "\n".join(lines)
    assert "?" in "\n".join(lines)
    assert len(lines) >= 2


def test_explicit_newlines_are_hard_breaks() -> None:
    text = "LINE ONE\nLINE TWO"
    lines = wrap_text(text, font_size=40, frame_width=1080, max_width_ratio=0.9)
    assert lines[0] == "LINE ONE"
    assert "LINE TWO" in lines


def test_cyrillic_not_rewritten() -> None:
    text = "Подожди… никто не говорит об этом"
    lines = wrap_text(text, font_size=48, frame_width=360, max_width_ratio=0.8)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "").replace("\n", "")
    assert "говорит" in "\n".join(lines)


def test_narrow_frame_produces_multiple_lines() -> None:
    text = "one two three four five six seven eight"
    lines = wrap_text(text, font_size=48, frame_width=200, max_width_ratio=0.7)
    assert len(lines) >= 2
    assert " ".join(lines) == text
