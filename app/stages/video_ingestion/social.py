"""Download Instagram Reels / TikTok videos via yt-dlp into a temporary local file."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.video_ingestion.exceptions import VideoUnavailableError
from app.stages.video_ingestion.validation import normalize_extension, validate_extension

logger = get_logger(__name__)

_INSTAGRAM_HOSTS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}
_REEL_PATH = re.compile(r"^/(reel|reels|p)/[^/]+/?", re.IGNORECASE)

_TIKTOK_HOSTS = {
    "tiktok.com",
    "www.tiktok.com",
    "m.tiktok.com",
    "vm.tiktok.com",
    "vt.tiktok.com",
}
_TIKTOK_VIDEO_PATH = re.compile(
    r"^/@[^/]+/video/\d+",
    re.IGNORECASE,
)


def is_instagram_reel_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in _INSTAGRAM_HOSTS:
        return False
    return bool(_REEL_PATH.match(parsed.path or ""))


def is_tiktok_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host in {"vm.tiktok.com", "vt.tiktok.com"}:
        # Short links: path is a share code like /ZMhxxx/
        path = (parsed.path or "").strip("/")
        return bool(path)
    if host not in _TIKTOK_HOSTS:
        return False
    path = parsed.path or ""
    if _TIKTOK_VIDEO_PATH.match(path):
        return True
    # Some mobile share paths still resolve through yt-dlp.
    return "/video/" in path.lower()


def is_social_video_url(url: str) -> bool:
    return is_instagram_reel_url(url) or is_tiktok_url(url)


def social_platform(url: str) -> str | None:
    if is_instagram_reel_url(url):
        return "instagram"
    if is_tiktok_url(url):
        return "tiktok"
    return None


async def download_instagram_reel(url: str, settings: Settings) -> tuple[Path, str]:
    return await download_social_video(url, settings, platform="instagram")


async def download_social_video(
    url: str,
    settings: Settings,
    *,
    platform: str | None = None,
) -> tuple[Path, str]:
    """Download a social video to a temp file. Caller must delete the path when done.

    Returns (path, extension) where extension includes the leading dot.
    """
    platform = platform or social_platform(url) or "social"
    label = platform.capitalize()

    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise VideoUnavailableError(
            f"{label} download requires yt-dlp",
            details={"hint": "pip install yt-dlp", "url": _safe_url(url)},
        ) from exc

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{platform}_"))
    outtmpl = str(tmp_dir / "source.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "mp4/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": settings.download_timeout_seconds,
        "retries": 2,
    }

    def _run() -> Path:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise VideoUnavailableError(
                    f"yt-dlp returned no metadata for {label} URL",
                    details={"url": _safe_url(url), "platform": platform},
                )
            requested = ydl.prepare_filename(info)
            path = Path(requested)
            if not path.is_file():
                matches = list(tmp_dir.glob("source.*"))
                if not matches:
                    raise VideoUnavailableError(
                        f"{label} download finished without a video file",
                        details={"url": _safe_url(url), "platform": platform},
                    )
                path = matches[0]
            return path

    try:
        path = await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=float(settings.download_timeout_seconds) + 30.0,
        )
    except asyncio.TimeoutError as exc:
        _cleanup_dir(tmp_dir)
        raise VideoUnavailableError(
            f"Timed out while downloading {label} video",
            details={"url": _safe_url(url), "platform": platform},
        ) from exc
    except VideoUnavailableError:
        _cleanup_dir(tmp_dir)
        raise
    except Exception as exc:  # noqa: BLE001
        _cleanup_dir(tmp_dir)
        raise VideoUnavailableError(
            f"Failed to download {label} video",
            details={"url": _safe_url(url), "platform": platform, "error": str(exc)},
        ) from exc

    extension = normalize_extension(path.name)
    try:
        validate_extension(extension, settings)
    except Exception:
        _cleanup_dir(tmp_dir)
        raise

    size = path.stat().st_size
    max_bytes = settings.max_upload_size_bytes
    if size > max_bytes:
        _cleanup_dir(tmp_dir)
        raise VideoUnavailableError(
            f"Downloaded {label} video exceeds size limit",
            details={"bytes": size, "max_bytes": max_bytes, "platform": platform},
        )
    if size == 0:
        _cleanup_dir(tmp_dir)
        raise VideoUnavailableError(
            f"Downloaded {label} video is empty",
            details={"url": _safe_url(url), "platform": platform},
        )

    logger.info(
        "Downloaded %s video url=%s path=%s bytes=%s",
        platform,
        _safe_url(url),
        path.name,
        size,
    )
    return path, extension


def _cleanup_dir(path: Path) -> None:
    try:
        for child in path.glob("*"):
            child.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        logger.warning("Failed to cleanup social download temp dir %s", path)


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
