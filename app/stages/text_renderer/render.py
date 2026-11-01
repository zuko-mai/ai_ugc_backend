"""FFmpeg burn-in and preview helpers for the text renderer."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.stages.text_renderer.ass import PreparedSegment, build_ass_document
from app.stages.text_renderer.exceptions import RenderError
from app.stages.text_renderer.fonts import (
    ass_font_name,
    emoji_ass_font_name,
    resolve_emoji_font_path,
    resolve_font_path,
    stage_fonts_dir,
)
from app.stages.text_renderer.schemas import TextStyle


@dataclass(frozen=True)
class RenderedBytes:
    video_bytes: bytes
    preview_jpeg: bytes | None
    extension: str


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RenderError("ffmpeg is not installed or not available on PATH")
    return ffmpeg


def _escape_filter_path(path: Path) -> str:
    # FFmpeg filtergraph path escaping for ass=/subtitles=.
    text = path.resolve().as_posix()
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


async def burn_text_video(
    source_bytes: bytes,
    *,
    style: TextStyle,
    segments: list[PreparedSegment],
    width: int,
    height: int,
    output_format: str,
    preview_at_seconds: float | None = None,
    source_suffix: str = ".mp4",
) -> RenderedBytes:
    ffmpeg = _require_ffmpeg()
    font_path = resolve_font_path(style.font_family, weight=style.font_weight)
    font_name = ass_font_name(font_path, style.font_family)
    emoji_font = resolve_emoji_font_path()
    emoji_name = emoji_ass_font_name(emoji_font)
    ass_body = build_ass_document(
        width=width,
        height=height,
        style=style,
        font_name=font_name,
        segments=segments,
        emoji_font_name=emoji_name,
    )
    ext = ".mov" if output_format == "mov" else ".mp4"

    def _run() -> RenderedBytes:
        with tempfile.TemporaryDirectory(prefix="textrender-") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / f"source{source_suffix}"
            ass_file = tmp_path / "overlay.ass"
            fonts_dir = tmp_path / "fonts"
            stage_fonts_dir(fonts_dir, text_font=font_path, emoji_font=emoji_font)
            dst = tmp_path / f"out{ext}"
            preview = tmp_path / "preview.jpg"

            src.write_bytes(source_bytes)
            ass_file.write_text(ass_body, encoding="utf-8")

            vf = (
                f"ass={_escape_filter_path(ass_file)}:"
                f"fontsdir={_escape_filter_path(fonts_dir)}"
            )
            video_cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(dst),
            ]
            result = subprocess.run(video_cmd, capture_output=True, check=False)
            if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                stderr = result.stderr.decode(errors="replace")[:800]
                raise RenderError(f"ffmpeg failed to burn text into the video: {stderr}")

            preview_bytes: bytes | None = None
            if preview_at_seconds is not None:
                preview_cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(0.0, preview_at_seconds):.3f}",
                    "-i",
                    str(dst),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(preview),
                ]
                preview_result = subprocess.run(preview_cmd, capture_output=True, check=False)
                if preview_result.returncode == 0 and preview.is_file() and preview.stat().st_size:
                    preview_bytes = preview.read_bytes()

            return RenderedBytes(
                video_bytes=dst.read_bytes(),
                preview_jpeg=preview_bytes,
                extension=ext,
            )

    return await asyncio.to_thread(_run)


async def export_without_text(
    source_bytes: bytes,
    *,
    output_format: str,
    source_suffix: str = ".mp4",
) -> RenderedBytes:
    """Copy/remux into a new file without burning text or mutating the source."""
    ffmpeg = _require_ffmpeg()
    ext = ".mov" if output_format == "mov" else ".mp4"

    def _run() -> RenderedBytes:
        with tempfile.TemporaryDirectory(prefix="textrender-copy-") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / f"source{source_suffix}"
            dst = tmp_path / f"out{ext}"
            src.write_bytes(source_bytes)
            # Prefer stream copy; fall back to a light remux encode if needed.
            copy_cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(dst),
            ]
            result = subprocess.run(copy_cmd, capture_output=True, check=False)
            if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                encode_cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ]
                result = subprocess.run(encode_cmd, capture_output=True, check=False)
                if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                    stderr = result.stderr.decode(errors="replace")[:800]
                    raise RenderError(f"ffmpeg failed to export video without text: {stderr}")
            return RenderedBytes(
                video_bytes=dst.read_bytes(),
                preview_jpeg=None,
                extension=ext,
            )

    return await asyncio.to_thread(_run)


async def render_preview_frame(
    source_bytes: bytes,
    *,
    style: TextStyle,
    segments: list[PreparedSegment],
    width: int,
    height: int,
    preview_at_seconds: float,
    source_suffix: str = ".mp4",
) -> bytes:
    ffmpeg = _require_ffmpeg()
    font_path = resolve_font_path(style.font_family, weight=style.font_weight)
    font_name = ass_font_name(font_path, style.font_family)
    emoji_font = resolve_emoji_font_path()
    emoji_name = emoji_ass_font_name(emoji_font)
    ass_body = build_ass_document(
        width=width,
        height=height,
        style=style,
        font_name=font_name,
        segments=segments,
        emoji_font_name=emoji_name,
    )

    def _run() -> bytes:
        with tempfile.TemporaryDirectory(prefix="textrender-preview-") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / f"source{source_suffix}"
            ass_file = tmp_path / "overlay.ass"
            fonts_dir = tmp_path / "fonts"
            stage_fonts_dir(fonts_dir, text_font=font_path, emoji_font=emoji_font)
            dst = tmp_path / "preview.jpg"
            src.write_bytes(source_bytes)
            ass_file.write_text(ass_body, encoding="utf-8")
            vf = (
                f"ass={_escape_filter_path(ass_file)}:"
                f"fontsdir={_escape_filter_path(fonts_dir)}"
            )
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, preview_at_seconds):.3f}",
                "-i",
                str(src),
                "-vf",
                vf,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(dst),
            ]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
                stderr = result.stderr.decode(errors="replace")[:800]
                raise RenderError(f"ffmpeg failed to render text preview: {stderr}")
            return dst.read_bytes()

    return await asyncio.to_thread(_run)


__all__ = [
    "RenderedBytes",
    "burn_text_video",
    "export_without_text",
    "render_preview_frame",
]
