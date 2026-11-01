"""Fetch, decode, and validate input images for face replacement."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.face_replacement.exceptions import (
    ImageDecodeError,
    ImageFetchError,
    ImageResolutionError,
    ImageTooLargeError,
)
from app.storage.base import FileStorage

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@dataclass(frozen=True)
class LoadedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    source_url: str


async def load_image(
    url: str,
    *,
    settings: Settings,
    storage: FileStorage,
    role: str,
    min_width: int,
    min_height: int,
) -> LoadedImage:
    """Load an image from a local /media URL or a remote HTTP(S) URL."""
    data, mime_hint = await _fetch_bytes(url, settings=settings, storage=storage)
    return _finalize_image(
        data,
        mime_hint=mime_hint,
        settings=settings,
        role=role,
        min_width=min_width,
        min_height=min_height,
        source_label=url,
    )


async def load_uploaded_image(
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    settings: Settings,
    role: str,
    min_width: int,
    min_height: int,
) -> LoadedImage:
    """Load and validate an image from a multipart file upload."""
    label = f"upload:{filename or 'face_reference'}"
    mime_hint = _normalize_upload_mime(content_type, filename)
    if mime_hint is None and content_type and content_type.lower() not in {
        "application/octet-stream",
        "binary/octet-stream",
        "",
    }:
        raise ImageFetchError(
            f"{role} upload must be an image (JPEG/PNG/WebP/BMP)",
            details={
                "role": role,
                "filename": filename,
                "content_type": content_type,
            },
        )
    return _finalize_image(
        data,
        mime_hint=mime_hint,
        settings=settings,
        role=role,
        min_width=min_width,
        min_height=min_height,
        source_label=label,
    )


def _finalize_image(
    data: bytes,
    *,
    mime_hint: str | None,
    settings: Settings,
    role: str,
    min_width: int,
    min_height: int,
    source_label: str,
) -> LoadedImage:
    _validate_size(len(data), settings=settings, role=role, url=source_label)
    mime_type, width, height = decode_image_meta(data, mime_hint=mime_hint)
    _validate_resolution(
        width=width,
        height=height,
        min_width=min_width,
        min_height=min_height,
        role=role,
        url=source_label,
    )
    logger.info(
        "Loaded %s image source=%s size=%sx%s bytes=%s mime=%s",
        role,
        _safe_url(source_label),
        width,
        height,
        len(data),
        mime_type,
    )
    return LoadedImage(
        data=data,
        mime_type=mime_type,
        width=width,
        height=height,
        source_url=source_label,
    )


def _normalize_upload_mime(content_type: str | None, filename: str | None) -> str | None:
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime.startswith("image/"):
            return mime
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in _MIME_BY_EXT:
            return _MIME_BY_EXT[ext]
    return None


def decode_image_meta(data: bytes, *, mime_hint: str | None = None) -> tuple[str, int, int]:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(
            "Could not decode image bytes",
            details={"bytes": len(data), "mime_hint": mime_hint},
        )
    height, width = image.shape[:2]
    mime = mime_hint or _guess_mime(data) or "image/jpeg"
    return mime, int(width), int(height)


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ImageDecodeError("Failed to encode PNG")
    return bytes(buf)


async def _fetch_bytes(
    url: str,
    *,
    settings: Settings,
    storage: FileStorage,
) -> tuple[bytes, str | None]:
    media_key = _media_key_from_url(url, settings.public_base_url)
    if media_key is not None:
        try:
            path = storage.get_local_path(media_key)
        except Exception as exc:  # noqa: BLE001
            raise ImageFetchError(
                "Local media image not found",
                details={"url": _safe_url(url), "key": media_key, "error": str(exc)},
            ) from exc
        return await _read_file(path), _mime_from_path(path)

    if url.startswith("/") and not url.startswith("//"):
        # Relative /media/... path against this app
        if url.startswith("/media/"):
            key = url[len("/media/") :]
            try:
                path = storage.get_local_path(key)
            except Exception as exc:  # noqa: BLE001
                raise ImageFetchError(
                    "Local media image not found",
                    details={"url": url, "key": key, "error": str(exc)},
                ) from exc
            return await _read_file(path), _mime_from_path(path)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ImageFetchError(
            "Unsupported image URL scheme",
            details={"url": _safe_url(url), "scheme": parsed.scheme},
        )

    timeout = httpx.Timeout(settings.download_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                raise ImageFetchError(
                    f"Image URL returned HTTP {response.status_code}",
                    details={"url": _safe_url(url), "status_code": response.status_code},
                )
            content = response.content
            content_type = response.headers.get("content-type")
            mime = None
            if content_type:
                mime = content_type.split(";")[0].strip().lower()
                if mime and not mime.startswith("image/"):
                    # Some CDNs omit accurate types; still try decode later.
                    mime = None
            return content, mime or _mime_from_url(url)
    except ImageFetchError:
        raise
    except httpx.TimeoutException as exc:
        raise ImageFetchError(
            "Timed out while downloading image",
            details={"url": _safe_url(url), "error": str(exc)},
        ) from exc
    except httpx.HTTPError as exc:
        raise ImageFetchError(
            "Failed to download image",
            details={"url": _safe_url(url), "error": str(exc)},
        ) from exc


async def _read_file(path: Path) -> bytes:
    import aiofiles

    async with aiofiles.open(path, "rb") as fh:
        return await fh.read()


def _validate_size(size: int, *, settings: Settings, role: str, url: str) -> None:
    max_bytes = settings.max_face_image_size_mb * 1024 * 1024
    if size > max_bytes:
        raise ImageTooLargeError(
            f"{role} image exceeds size limit ({settings.max_face_image_size_mb} MB)",
            details={
                "role": role,
                "bytes": size,
                "max_bytes": max_bytes,
                "url": _safe_url(url),
            },
        )
    if size <= 0:
        raise ImageFetchError(
            f"{role} image is empty",
            details={"role": role, "url": _safe_url(url)},
        )


def _validate_resolution(
    *,
    width: int,
    height: int,
    min_width: int,
    min_height: int,
    role: str,
    url: str,
) -> None:
    if width < min_width or height < min_height:
        raise ImageResolutionError(
            f"{role} image resolution {width}x{height} is below minimum {min_width}x{min_height}",
            details={
                "role": role,
                "width": width,
                "height": height,
                "min_width": min_width,
                "min_height": min_height,
                "url": _safe_url(url),
            },
        )


def _media_key_from_url(url: str, public_base_url: str) -> str | None:
    base = public_base_url.rstrip("/")
    prefixes = (f"{base}/media/", "http://testserver/media/", "http://localhost:8000/media/")
    for prefix in prefixes:
        if url.startswith(prefix):
            return url[len(prefix) :]
    return None


def _mime_from_path(path: Path) -> str | None:
    return _MIME_BY_EXT.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]


def _mime_from_url(url: str) -> str | None:
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    return mimetypes.guess_type(path)[0]


def _guess_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
