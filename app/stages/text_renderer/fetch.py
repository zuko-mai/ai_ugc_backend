"""Read video bytes from local /media URLs or remote HTTP(S) URLs."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.stages.text_renderer.exceptions import VideoFetchError, VideoTooLargeError
from app.storage.base import FileStorage

_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/mp4",
}


async def read_video_bytes(
    url: str,
    *,
    settings: Settings,
    storage: FileStorage,
    max_bytes: int,
) -> tuple[bytes, str | None]:
    key = media_key_from_url(url, settings.public_base_url)
    if key is not None:
        _assert_safe_media_key(key)
        try:
            path = storage.get_local_path(key)
        except Exception as exc:  # noqa: BLE001
            raise VideoFetchError(
                "Stored media object was not found",
                details={"url": _safe_url(url), "key": key, "error": str(exc)},
            ) from exc
        data = path.read_bytes()
        _check_size(data, max_bytes=max_bytes, url=url)
        return data, _MIME_BY_EXT.get(path.suffix.lower())

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise VideoFetchError(
            "Unsupported URL scheme",
            details={"url": _safe_url(url), "scheme": parsed.scheme or ""},
        )

    timeout = httpx.Timeout(settings.download_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise VideoFetchError(
            "Timed out while downloading the video",
            details={"url": _safe_url(url)},
        ) from exc
    except httpx.HTTPError as exc:
        raise VideoFetchError(
            "Failed to download the video",
            details={"url": _safe_url(url), "error": str(exc)},
        ) from exc

    if response.status_code >= 400:
        raise VideoFetchError(
            f"URL returned HTTP {response.status_code}",
            details={"url": _safe_url(url), "status_code": response.status_code},
        )
    data = response.content
    _check_size(data, max_bytes=max_bytes, url=url)
    header = response.headers.get("content-type")
    mime = header.split(";")[0].strip().lower() if header else None
    return data, mime


def media_key_from_url(url: str, public_base_url: str) -> str | None:
    base = public_base_url.rstrip("/")
    prefix = f"{base}/media/"
    if url.startswith(prefix):
        return url[len(prefix) :]
    if url.startswith("/media/"):
        return url[len("/media/") :]
    return None


def _assert_safe_media_key(key: str) -> None:
    cleaned = key.replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise VideoFetchError(
            "Invalid media path",
            details={"key": key},
        )


def _check_size(data: bytes, *, max_bytes: int, url: str) -> None:
    if not data:
        raise VideoFetchError("Downloaded video is empty", details={"url": _safe_url(url)})
    if len(data) > max_bytes:
        raise VideoTooLargeError(
            "Downloaded video exceeds the size limit",
            details={"url": _safe_url(url), "bytes": len(data), "max_bytes": max_bytes},
        )


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
