from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.stages.video_ingestion.exceptions import VideoTooLargeError, VideoUnavailableError
from app.stages.video_ingestion.validation import (
    extension_from_url,
    normalize_extension,
    validate_extension,
    validate_mime_type,
    validate_size,
)

logger = get_logger(__name__)


async def stream_from_url(
    url: str,
    settings: Settings,
) -> tuple[str, str | None, AsyncIterator[bytes]]:
    """Stream a remote video; callers must fully consume the iterator."""
    timeout = httpx.Timeout(settings.download_timeout_seconds)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    response: httpx.Response | None = None

    try:
        request = client.build_request("GET", url)
        response = await client.send(request, stream=True)

        if response.status_code >= 400:
            raise VideoUnavailableError(
                f"Remote video URL returned HTTP {response.status_code}",
                details={"url": _safe_url(url), "status_code": response.status_code},
            )

        content_type = response.headers.get("content-type")
        content_length = response.headers.get("content-length")
        if content_length:
            validate_size(int(content_length), settings)

        validate_mime_type(content_type, settings)

        extension = extension_from_url(url)
        content_disp = response.headers.get("content-disposition", "")
        if not extension and "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip().strip("\"'")
            extension = normalize_extension(filename)
        validate_extension(extension, settings)

    except (VideoUnavailableError, VideoTooLargeError):
        await _close(response, client)
        raise
    except httpx.HTTPError as exc:
        await _close(response, client)
        raise VideoUnavailableError(
            "Failed to download video from URL",
            details={"url": _safe_url(url), "error": str(exc)},
        ) from exc
    except Exception:
        await _close(response, client)
        raise

    assert response is not None

    async def _iter() -> AsyncIterator[bytes]:
        total = 0
        try:
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                total += len(chunk)
                validate_size(total, settings)
                yield chunk
            logger.info(
                "Downloaded remote video url=%s bytes=%s content_type=%s",
                _safe_url(url),
                total,
                content_type,
            )
        except VideoTooLargeError:
            raise
        except httpx.HTTPError as exc:
            raise VideoUnavailableError(
                "Error while streaming remote video",
                details={"url": _safe_url(url), "error": str(exc)},
            ) from exc
        finally:
            await _close(response, client)

    return extension, content_type, _iter()


async def _close(response: httpx.Response | None, client: httpx.AsyncClient) -> None:
    if response is not None:
        await response.aclose()
    await client.aclose()


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
