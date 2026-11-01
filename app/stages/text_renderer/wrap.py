"""Deterministic word wrapping that preserves original wording."""

from __future__ import annotations


def wrap_text(
    text: str,
    *,
    font_size: int,
    frame_width: int,
    max_width_ratio: float = 0.86,
) -> list[str]:
    """Wrap `text` into visual lines without changing words or punctuation.

    Explicit newlines from the user are kept as hard breaks. Soft wraps are
    inserted only between whitespace-separated tokens when a line would exceed
    the available width. Characters themselves are never rewritten.
    """
    if not text:
        return []

    max_width_px = max(32, int(frame_width * max_width_ratio))
    # Average glyph advance for Latin/Cyrillic sans at typical UGC sizes.
    avg_advance = max(6.0, font_size * 0.55)
    max_chars = max(8, int(max_width_px / avg_advance))

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        lines.extend(_wrap_paragraph(paragraph, max_chars=max_chars))
    return lines if lines else [""]


def _wrap_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    tokens = paragraph.split(" ")
    if len(tokens) == 1 and len(tokens[0]) <= max_chars:
        return [paragraph]

    lines: list[str] = []
    current = ""
    for token in tokens:
        if not token:
            # Preserve multiple spaces as empty tokens by keeping a space.
            candidate = f"{current} " if current else " "
        elif not current:
            candidate = token
        else:
            candidate = f"{current} {token}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        # Oversized single token: hard-split without altering characters.
        if len(token) > max_chars:
            lines.extend(_hard_split(token, max_chars=max_chars))
        else:
            current = token

    if current:
        lines.append(current)
    elif not lines:
        lines.append("")
    return lines


def _hard_split(token: str, *, max_chars: int) -> list[str]:
    return [token[i : i + max_chars] for i in range(0, len(token), max_chars)]


def estimate_block_height(
    line_count: int,
    *,
    font_size: int,
    line_spacing: float,
) -> int:
    if line_count <= 0:
        return 0
    line_height = font_size * line_spacing
    return int(round(line_height * line_count))
