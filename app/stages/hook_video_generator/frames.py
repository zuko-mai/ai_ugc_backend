"""Load and decode the image-to-video first frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.stages.hook_video_generator.exceptions import (
    FrameDecodeError,
    FrameFetchError,
    FrameTooLargeError,
    PayloadFetchError,
)
from app.stages.hook_video_generator.fetch import read_url_bytes
from app.storage.base import FileStorage

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@dataclass(frozen=True)
class LoadedFrame:
    data: bytes
    mime_type: str
    width: int
    height: int
    source_url: str


async def load_frame(
    url: str,
    *,
    settings: Settings,
    storage: FileStorage,
) -> LoadedFrame:
    max_bytes = settings.max_hook_frame_size_mb * 1024 * 1024
    try:
        data, mime_hint = await read_url_bytes(
            url,
            settings=settings,
            storage=storage,
            max_bytes=max_bytes,
        )
    except PayloadFetchError as exc:
        if "max_bytes" in exc.details:
            raise FrameTooLargeError(exc.message, details=exc.details) from exc
        raise FrameFetchError(exc.message, details=exc.details) from exc
    return decode_frame(data, source_url=url, mime_hint=mime_hint)


async def load_archived_frame(
    *,
    storage: FileStorage,
    key: str,
    source_url: str,
) -> LoadedFrame:
    try:
        path = storage.get_local_path(key)
    except Exception as exc:  # noqa: BLE001
        raise FrameFetchError(
            "Archived first frame is missing",
            details={"key": key, "error": str(exc)},
        ) from exc
    data = path.read_bytes()
    if not data:
        raise FrameFetchError("Archived first frame is empty", details={"key": key})
    mime_hint = _MIME_BY_EXT.get(Path(key).suffix.lower())
    return decode_frame(data, source_url=source_url, mime_hint=mime_hint)


def decode_frame(data: bytes, *, source_url: str, mime_hint: str | None) -> LoadedFrame:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise FrameDecodeError(
            "First frame is not a decodable image",
            details={"source_url": source_url},
        )
    height, width = image.shape[:2]
    if width < 1 or height < 1:
        raise FrameDecodeError(
            "First frame has no visible pixels",
            details={"source_url": source_url},
        )
    mime = mime_hint if mime_hint and mime_hint.startswith("image/") else _guess_mime(data)
    return LoadedFrame(
        data=data,
        mime_type=mime or "image/jpeg",
        width=int(width),
        height=int(height),
        source_url=source_url,
    )


def resolution_warnings(
    *,
    frame_width: int,
    frame_height: int,
    output_width: int,
    output_height: int,
) -> list[str]:
    if frame_width < output_width or frame_height < output_height:
        return [
            "first_frame_upscaled: "
            f"source {frame_width}x{frame_height} is smaller than output "
            f"{output_width}x{output_height}"
        ]
    return []


def _guess_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
