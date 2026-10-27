from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.video_ingestion.exceptions import UnsupportedFormatError, VideoTooLargeError

logger = get_logger(__name__)

_SIGNATURES: dict[str, list[bytes]] = {
    ".mp4": [b"ftyp"],
    ".mov": [b"ftyp"],
}


def normalize_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def extension_from_url(url: str) -> str:
    return normalize_extension(urlparse(url).path)


def validate_extension(extension: str, settings: Settings) -> None:
    if extension not in settings.extension_set:
        raise UnsupportedFormatError(
            f"Unsupported video format '{extension or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(settings.extension_set))}",
            details={"extension": extension, "allowed": sorted(settings.extension_set)},
        )


def validate_mime_type(content_type: str | None, settings: Settings) -> None:
    if not content_type:
        return
    mime = content_type.split(";")[0].strip().lower()
    if mime in {"application/octet-stream", "binary/octet-stream"}:
        return
    if mime not in settings.mime_type_set:
        raise UnsupportedFormatError(
            f"Unsupported content type '{mime}'. "
            f"Allowed: {', '.join(sorted(settings.mime_type_set))}",
            details={"content_type": mime, "allowed": sorted(settings.mime_type_set)},
        )


def validate_size(size_bytes: int, settings: Settings) -> None:
    if size_bytes > settings.max_upload_size_bytes:
        raise VideoTooLargeError(
            f"Video exceeds size limit of {settings.max_upload_size_mb} MB",
            details={
                "size_bytes": size_bytes,
                "max_bytes": settings.max_upload_size_bytes,
            },
        )


def validate_container_magic(header: bytes, extension: str) -> None:
    expected = _SIGNATURES.get(extension)
    if not expected:
        return
    marker = header[4:8] if len(header) >= 8 else b""
    if marker not in expected and b"ftyp" not in header[:64]:
        raise UnsupportedFormatError(
            "File content does not look like a valid MP4/MOV container",
            details={"extension": extension},
        )
    logger.debug("Container magic validated for extension=%s", extension)
