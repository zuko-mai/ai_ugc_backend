import re
import uuid
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import VideoNotFoundError
from app.storage.base import FileStorage

logger = get_logger(__name__)

_ORIGINAL_KEY_RE = re.compile(
    r"^originals/(?P<video_id>[0-9a-fA-F-]{36})/source(?P<ext>\.[A-Za-z0-9]+)$"
)


async def resolve_video_ref(
    *,
    video_id: UUID | None,
    video_url: str | None,
    storage: FileStorage,
    settings: Settings,
) -> tuple[UUID, Path]:
    """Resolve video by id and/or url. At least one must be provided.

    - video_id only: look in storage originals / hook source_cache
    - video_url only: local media (extract id from originals/...) or download + derived id
    - both: load from url, keep provided video_id for cache keys
    """
    if video_id is None and not video_url:
        raise VideoNotFoundError("Provide video_id or video_url")

    if video_url:
        resolved_id = video_id or _video_id_from_url(video_url, settings)
        path = await _resolve_from_url(
            video_id=resolved_id,
            video_url=video_url,
            storage=storage,
            settings=settings,
        )
        return resolved_id, path

    assert video_id is not None
    path = await _resolve_from_id(video_id=video_id, storage=storage)
    return video_id, path


async def _resolve_from_id(*, video_id: UUID, storage: FileStorage) -> Path:
    for key in (
        f"originals/{video_id}/source.mp4",
        f"originals/{video_id}/source.mov",
        f"hooks/{video_id}/source_cache.mp4",
        f"hooks/{video_id}/source_cache.mov",
    ):
        if await storage.exists(key):
            path = storage.get_local_path(key)
            logger.info("Resolved video by id=%s key=%s", video_id, key)
            return path

    raise VideoNotFoundError(
        "No cached/original video found for video_id",
        details={
            "video_id": str(video_id),
            "looked_for": [
                f"originals/{video_id}/source.mp4|.mov",
                f"hooks/{video_id}/source_cache.mp4|.mov",
            ],
        },
    )


async def _resolve_from_url(
    *,
    video_id: UUID,
    video_url: str,
    storage: FileStorage,
    settings: Settings,
) -> Path:
    media_key = _media_key_from_url(video_url, settings)
    if media_key is not None:
        try:
            path = storage.get_local_path(media_key)
            logger.info("Resolved local media key=%s path=%s", media_key, path)
            return path
        except Exception as exc:
            raise VideoNotFoundError(
                "video_url points to missing local media",
                details={"video_url": video_url, "key": media_key, "error": str(exc)},
            ) from exc

    suffix = Path(urlparse(video_url).path).suffix or ".mp4"
    cache_key = f"hooks/{video_id}/source_cache{suffix}"
    if await storage.exists(cache_key):
        return storage.get_local_path(cache_key)

    logger.info("Downloading remote video for hook analysis video_id=%s", video_id)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.download_timeout_seconds),
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", video_url) as response:
                if response.status_code >= 400:
                    raise VideoNotFoundError(
                        f"Remote video returned HTTP {response.status_code}",
                        details={"video_url": _safe_url(video_url)},
                    )

                async def _stream():
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        yield chunk

                await storage.save_stream(
                    cache_key,
                    _stream(),
                    content_type=response.headers.get("content-type"),
                )
    except VideoNotFoundError:
        raise
    except Exception as exc:
        raise VideoNotFoundError(
            "Failed to download video for hook analysis",
            details={"video_url": _safe_url(video_url), "error": str(exc)},
        ) from exc

    return storage.get_local_path(cache_key)


def _video_id_from_url(video_url: str, settings: Settings) -> UUID:
    media_key = _media_key_from_url(video_url, settings)
    if media_key:
        match = _ORIGINAL_KEY_RE.match(media_key)
        if match:
            return UUID(match.group("video_id"))
    # Stable id for arbitrary remote URLs so cache folders stay consistent.
    return uuid.uuid5(uuid.NAMESPACE_URL, video_url)


def _media_key_from_url(video_url: str, settings: Settings) -> str | None:
    base = settings.public_base_url.rstrip("/")
    prefixes = (
        f"{base}/media/",
        "/media/",
        "http://testserver/media/",
        "http://localhost:8000/media/",
    )
    for prefix in prefixes:
        if video_url.startswith(prefix):
            return video_url[len(prefix) :].lstrip("/")
    return None


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
