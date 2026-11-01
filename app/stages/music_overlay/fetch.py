"""Read video or music bytes from local /media URLs or remote HTTP(S) URLs."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.stages.music_overlay.exceptions import MediaFetchError, MediaTooLargeError
from app.storage.base import FileStorage

_AUDIO_MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".m4b": "audio/mp4",
    ".caf": "audio/x-caf",
    ".wma": "audio/x-ms-wma",
}
_VIDEO_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


async def read_media_bytes(
    url: str,
    *,
    settings: Settings,
    storage: FileStorage,
    max_bytes: int,
    role: str,
) -> tuple[bytes, str | None]:
    key = media_key_from_url(url, settings.public_base_url)
    if key is not None:
        _assert_safe_media_key(key, role=role)
        try:
            path = storage.get_local_path(key)
        except Exception as exc:  # noqa: BLE001
            raise MediaFetchError(
                f"Stored {role} file was not found",
                details={"role": role, "url": _safe_url(url), "key": key, "error": str(exc)},
            ) from exc
        data = path.read_bytes()
        _check_size(data, max_bytes=max_bytes, url=url, role=role)
        mime = _AUDIO_MIME_BY_EXT.get(path.suffix.lower()) or _VIDEO_MIME_BY_EXT.get(
            path.suffix.lower()
        )
        return data, mime

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise MediaFetchError(
            f"Unsupported URL scheme for {role}",
            details={"role": role, "url": _safe_url(url), "scheme": parsed.scheme or ""},
        )

    timeout = httpx.Timeout(settings.download_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise MediaFetchError(
            f"Timed out while downloading the {role}",
            details={"role": role, "url": _safe_url(url)},
        ) from exc
    except httpx.HTTPError as exc:
        raise MediaFetchError(
            f"Failed to download the {role}",
            details={"role": role, "url": _safe_url(url), "error": str(exc)},
        ) from exc

    if response.status_code >= 400:
        raise MediaFetchError(
            f"{role.capitalize()} URL returned HTTP {response.status_code}",
            details={
                "role": role,
                "url": _safe_url(url),
                "status_code": response.status_code,
            },
        )
    data = response.content
    _check_size(data, max_bytes=max_bytes, url=url, role=role)
    header = response.headers.get("content-type")
    mime = header.split(";")[0].strip().lower() if header else None
    return data, mime


def media_key_from_url(url: str, public_base_url: str) -> str | None:
    base = public_base_url.rstrip("/")
    prefix = f"{base}/media/"
    if url.startswith(prefix):
        return url[len(prefix) :].split("?", 1)[0]
    if url.startswith("/media/"):
        return url[len("/media/") :].split("?", 1)[0]
    return None


def suffix_from_url(url: str, mime: str | None, *, kind: str) -> str:
    path = url.lower().split("?", 1)[0]
    for ext in (*_AUDIO_MIME_BY_EXT, *_VIDEO_MIME_BY_EXT):
        if path.endswith(ext):
            return ext
    if mime:
        lowered = mime.lower()
        if "mpeg" in lowered or lowered.endswith("/mp3"):
            return ".mp3"
        if "wav" in lowered:
            return ".wav"
        if "flac" in lowered:
            return ".flac"
        if "ogg" in lowered or "opus" in lowered:
            return ".ogg"
        if "aac" in lowered:
            return ".aac"
        if "mp4" in lowered or "m4a" in lowered:
            return ".m4a" if kind == "music" else ".mp4"
        if "quicktime" in lowered:
            return ".mov"
    return ".mp3" if kind == "music" else ".mp4"


def _assert_safe_media_key(key: str, *, role: str) -> None:
    cleaned = key.replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise MediaFetchError(
            f"Invalid {role} media path",
            details={"role": role, "key": key},
        )


def _check_size(data: bytes, *, max_bytes: int, url: str, role: str) -> None:
    if not data:
        raise MediaFetchError(
            f"Downloaded {role} file is empty",
            details={"role": role, "url": _safe_url(url)},
        )
    if len(data) > max_bytes:
        raise MediaTooLargeError(
            f"Downloaded {role} file exceeds the size limit",
            details={
                "role": role,
                "url": _safe_url(url),
                "bytes": len(data),
                "max_bytes": max_bytes,
            },
        )


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
