"""Wrap emoji runs so libass uses an emoji-capable fallback font.

libass does not reliably fall back from Arial/.notdef to an emoji font in
fontsdir, and Apple Color Emoji (sbix) is not usable via libass. Monochrome
Noto Emoji works when emoji spans are tagged with {\\fnNoto Emoji}.
"""

from __future__ import annotations

import re

DEFAULT_EMOJI_FONT_NAME = "Noto Emoji"

# Flags, keycaps, and emoji + ZWJ / VS16 / skin-tone sequences.
_EMOJI_RUN = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"
    r"|"
    r"[0-9#*]\uFE0F?\u20E3"
    r"|"
    r"(?:"
    r"[\U00002600-\U000026FF\U00002700-\U000027BF"
    r"\U0001F300-\U0001FAFF"
    r"]"
    r"|\uFE0F"
    r"|[\U0001F3FB-\U0001F3FF]"
    r")+"
    r"(?:\u200D"
    r"(?:"
    r"[\U00002600-\U000026FF\U00002700-\U000027BF\U0001F300-\U0001FAFF]"
    r"|\uFE0F"
    r"|[\U0001F3FB-\U0001F3FF]"
    r")+"
    r")*"
    r")"
)


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RUN.search(text or ""))


def wrap_emoji_for_ass(text: str, *, emoji_font_name: str = DEFAULT_EMOJI_FONT_NAME) -> str:
    """Escape ASS specials in plain runs; tag emoji runs with \\fn overrides."""
    if not text:
        return ""
    if not contains_emoji(text):
        return _escape_plain(text)

    parts: list[str] = []
    last = 0
    for match in _EMOJI_RUN.finditer(text):
        if match.start() > last:
            parts.append(_escape_plain(text[last : match.start()]))
        # Override tags must stay unescaped; emoji codepoints pass through.
        parts.append(f"{{\\fn{emoji_font_name}}}{match.group(0)}{{\\r}}")
        last = match.end()
    if last < len(text):
        parts.append(_escape_plain(text[last:]))
    return "".join(parts)


def _escape_plain(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
