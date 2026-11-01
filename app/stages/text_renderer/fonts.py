"""Resolve a system font file for deterministic FFmpeg/libass burn-in."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from app.stages.text_renderer.emoji_ass import DEFAULT_EMOJI_FONT_NAME
from app.stages.text_renderer.exceptions import FontNotFoundError

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_EMOJI_FALLBACK_PATHS = (
    _ASSETS_DIR / "NotoEmoji.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf"),
    Path("/usr/share/fonts/noto/NotoEmoji-Regular.ttf"),
    Path("/usr/local/share/fonts/NotoEmoji-Regular.ttf"),
)

# Prefer fonts with broad Latin + Cyrillic coverage.
_FAMILY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "arial": (
        "Arial Unicode MS",
        "Arial Unicode",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
    ),
    "helvetica": ("Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"),
    "dejavu sans": ("DejaVu Sans", "Arial Unicode MS", "Noto Sans"),
    "noto sans": ("Noto Sans", "DejaVu Sans", "Arial Unicode MS"),
    "liberation sans": ("Liberation Sans", "DejaVu Sans", "Arial"),
    "roboto": ("Roboto", "Noto Sans", "DejaVu Sans", "Arial"),
}

_BOLD_QUERY: dict[str, tuple[str, ...]] = {
    "arial": ("Arial Bold", "Arial-BoldMT", "Helvetica Bold", "DejaVu Sans Bold"),
    "helvetica": ("Helvetica Bold", "Arial Bold", "DejaVu Sans Bold"),
    "dejavu sans": ("DejaVu Sans Bold", "Arial Bold", "Noto Sans Bold"),
    "noto sans": ("Noto Sans Bold", "DejaVu Sans Bold", "Arial Bold"),
    "liberation sans": ("Liberation Sans Bold", "DejaVu Sans Bold", "Arial Bold"),
    "roboto": ("Roboto Bold", "Noto Sans Bold", "DejaVu Sans Bold", "Arial Bold"),
}

_BLACK_QUERY: dict[str, tuple[str, ...]] = {
    "arial": ("Arial Black", "Arial Bold", "Helvetica Bold", "DejaVu Sans Bold"),
    "helvetica": ("Arial Black", "Helvetica Bold", "Arial Bold"),
    "dejavu sans": ("DejaVu Sans Bold", "Arial Black", "Arial Bold"),
    "noto sans": ("Noto Sans Black", "Noto Sans Bold", "Arial Black"),
    "liberation sans": ("Liberation Sans Bold", "Arial Black", "Arial Bold"),
    "roboto": ("Roboto Black", "Roboto Bold", "Arial Black", "Arial Bold"),
}

_FALLBACK_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/local/share/fonts/DejaVuSans.ttf"),
)

_BOLD_FALLBACK_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Black.ttf"),
    Path("/Library/Fonts/Arial Black.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
)

_BLACK_FALLBACK_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial Black.ttf"),
    Path("/Library/Fonts/Arial Black.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
)


@lru_cache(maxsize=64)
def resolve_font_path(font_family: str, *, weight: str = "regular") -> Path:
    """Return an absolute .ttf/.otf path for the requested family and weight."""
    key = font_family.strip().lower()
    weight_key = (weight or "regular").strip().lower()
    if weight_key not in {"regular", "bold", "black"}:
        weight_key = "regular"

    if weight_key == "black":
        queries = _BLACK_QUERY.get(key, (f"{font_family.strip()} Black", "Arial Black", "Arial Bold"))
        path_fallbacks = _BLACK_FALLBACK_PATHS
    elif weight_key == "bold":
        queries = _BOLD_QUERY.get(key, (f"{font_family.strip()} Bold", "Arial Bold", "DejaVu Sans Bold"))
        path_fallbacks = _BOLD_FALLBACK_PATHS
    else:
        queries = _FAMILY_CANDIDATES.get(
            key, (font_family.strip(),) + _FAMILY_CANDIDATES["arial"]
        )
        path_fallbacks = _FALLBACK_PATHS

    for family in queries:
        path = _fc_match(family)
        if path is not None and _looks_like_weight(path, weight_key):
            return path
        # fc-match can lie (e.g. Arial Bold → Verdana); try exact filename next.
        path = _direct_named_file(family)
        if path is not None:
            return path

    for candidate in path_fallbacks:
        if candidate.is_file():
            return candidate.resolve()

    # Last resort: regular face (ASS Bold flag can still thicken slightly).
    for family in _FAMILY_CANDIDATES.get(key, (font_family.strip(), "Arial")):
        path = _fc_match(family)
        if path is not None:
            return path
    for candidate in _FALLBACK_PATHS:
        if candidate.is_file():
            return candidate.resolve()

    raise FontNotFoundError(
        "No usable TrueType/OpenType font found for text rendering",
        details={"font_family": font_family, "font_weight": weight_key},
    )


def fonts_directory(font_path: Path) -> Path:
    return font_path.resolve().parent


@lru_cache(maxsize=8)
def resolve_emoji_font_path() -> Path | None:
    """Monochrome Noto Emoji for libass (Apple Color Emoji / sbix is unsupported)."""
    for candidate in _EMOJI_FALLBACK_PATHS:
        if candidate.is_file():
            return candidate.resolve()
    for family in (DEFAULT_EMOJI_FONT_NAME, "Noto Color Emoji", "Segoe UI Emoji"):
        path = _fc_match(family)
        if path is not None and path.suffix.lower() in {".ttf", ".otf"}:
            # Prefer outline/emoji fonts; skip Apple Color Emoji .ttc (sbix).
            name = path.name.lower()
            if "color emoji" in name or path.suffix.lower() == ".ttc":
                continue
            return path
    return None


def emoji_ass_font_name(emoji_font_path: Path | None) -> str | None:
    if emoji_font_path is None:
        return None
    return DEFAULT_EMOJI_FONT_NAME


def stage_fonts_dir(fonts_dir: Path, *, text_font: Path, emoji_font: Path | None) -> None:
    """Copy text + emoji faces into an isolated fontsdir for the ass filter."""
    fonts_dir.mkdir(parents=True, exist_ok=True)
    local_font = fonts_dir / text_font.name
    local_font.write_bytes(text_font.read_bytes())
    if emoji_font is not None and emoji_font.is_file():
        dest = fonts_dir / emoji_font.name
        if dest.resolve() != local_font.resolve():
            dest.write_bytes(emoji_font.read_bytes())


def ass_font_name(font_path: Path, requested_family: str) -> str:
    """Name written into the ASS Style line (libass looks up by family)."""
    if requested_family.strip():
        # Keep the base family; weight is carried by the file + Bold flag.
        return requested_family.strip()
    return font_path.stem.replace("-", " ").replace("_", " ")


def _direct_named_file(family: str) -> Path | None:
    """Resolve well-known macOS/Linux font files without trusting fc-match."""
    normalized = family.strip().lower().replace("-", " ")
    mapping = {
        "arial": Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        "arial bold": Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        "arial black": Path("/System/Library/Fonts/Supplemental/Arial Black.ttf"),
        "arial unicode": Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        "arial unicode ms": Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        "dejavu sans": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        "dejavu sans bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    }
    path = mapping.get(normalized)
    if path is not None and path.is_file():
        return path.resolve()
    # Also try /Library/Fonts variants on macOS.
    alt = Path("/Library/Fonts") / f"{family.strip()}.ttf"
    if alt.is_file():
        return alt.resolve()
    return None


def _looks_like_weight(path: Path, weight: str) -> bool:
    name = path.stem.lower().replace("-", " ").replace("_", " ")
    if weight == "regular":
        return "bold" not in name and "black" not in name
    if weight == "black":
        return "black" in name or "heavy" in name or "bold" in name
    if weight == "bold":
        return "bold" in name or "black" in name or "heavy" in name
    return True


def _fc_match(family: str) -> Path | None:
    fc = shutil.which("fc-match")
    if not fc:
        return None
    try:
        result = subprocess.run(
            [fc, "-f", "%{file}", family],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    path_str = (result.stdout or "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
        return path.resolve()
    return None
