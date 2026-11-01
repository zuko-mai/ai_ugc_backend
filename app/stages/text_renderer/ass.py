"""Build ASS subtitle scripts for deterministic burn-in (Unicode + timing)."""

from __future__ import annotations

from dataclasses import dataclass

from app.stages.text_renderer.emoji_ass import wrap_emoji_for_ass
from app.stages.text_renderer.schemas import TextSegment, TextStyle
from app.stages.text_renderer.wrap import wrap_text


@dataclass(frozen=True)
class PreparedSegment:
    start_seconds: float
    end_seconds: float
    original_text: str
    lines: list[str]


def prepare_segments(
    *,
    text: str | None,
    segments: list[TextSegment],
    style: TextStyle,
    frame_width: int,
    duration_seconds: float,
) -> list[PreparedSegment]:
    """Normalize timing windows; wrap lines without rewriting user text."""
    full_end = max(0.05, float(duration_seconds) if duration_seconds > 0 else 3.0)
    source: list[tuple[float, float | None, str]]
    if segments:
        source = [(s.start_seconds, s.end_seconds, s.text) for s in segments]
    elif text:
        source = [(0.0, None, text)]
    else:
        return []

    prepared: list[PreparedSegment] = []
    for start, end, raw in source:
        # null end_seconds => hold until the real video ends (any duration).
        resolved_end = full_end if end is None else end
        clipped_end = resolved_end
        if duration_seconds > 0:
            if start >= duration_seconds:
                continue
            clipped_end = min(resolved_end, duration_seconds)
        if clipped_end <= start:
            continue
        lines = wrap_text(
            raw,
            font_size=style.font_size,
            frame_width=frame_width,
            max_width_ratio=style.max_width_ratio,
        )
        prepared.append(
            PreparedSegment(
                start_seconds=start,
                end_seconds=clipped_end,
                original_text=raw,
                lines=lines,
            )
        )
    return prepared


def build_ass_document(
    *,
    width: int,
    height: int,
    style: TextStyle,
    font_name: str,
    segments: list[PreparedSegment],
    emoji_font_name: str | None = None,
) -> str:
    """Return an ASS script. Dialogue text is escaped, never paraphrased."""
    # Top-row alignments so MarginV is measured from the top of the frame.
    alignment = _ass_alignment(style.text_align)
    primary = _ass_color(style.font_color)
    outline = _ass_color(style.outline_color)
    outline_w = max(0, int(round(style.outline_width)))
    margin_v = max(0, int(round(height * (style.padding_top / 100.0))))
    margin_h = max(8, int(round(width * (1.0 - style.max_width_ratio) / 2)))
    fontsize = style.font_size
    # ASS: -1 = bold on. Also pair with a real bold/black face when available.
    bold_flag = -1 if style.font_weight in {"bold", "black"} else 0

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{fontsize},{primary},&H000000FF,"
        f"{outline},&H64000000,{bold_flag},0,0,0,100,100,0,0,1,{outline_w},0,"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for segment in segments:
        body = "\\N".join(
            _format_ass_line(line, emoji_font_name=emoji_font_name)
            for line in segment.lines
        )
        # Preserve blank lines the user intentionally inserted.
        if not body and segment.original_text == "":
            continue
        events.append(
            "Dialogue: 0,"
            f"{_ass_time(segment.start_seconds)},{_ass_time(segment.end_seconds)},"
            f"Default,,0,0,0,,{body}"
        )
    return header + "\n".join(events) + ("\n" if events else "")


def _format_ass_line(text: str, *, emoji_font_name: str | None) -> str:
    if emoji_font_name:
        return wrap_emoji_for_ass(text, emoji_font_name=emoji_font_name)
    return _escape_ass_text(text)


def _ass_alignment(text_align: str) -> int:
    # Top row (7–9): MarginV = padding from the top edge of PlayResY.
    return {"left": 7, "center": 8, "right": 9}[text_align]


def _ass_color(hex_color: str) -> str:
    # ASS uses &HAABBGGRR
    value = hex_color.lstrip("#")
    r = value[0:2]
    g = value[2:4]
    b = value[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    # Escape override markers / backslashes; keep Unicode glyphs intact.
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
