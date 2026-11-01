from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.music_overlay.exceptions import (
    InvalidSettingsError,
    JobNotFoundError,
    MediaFetchError,
    MediaTooLargeError,
    MusicOverlayError,
    OverlayError,
    ProbeError,
    UnsupportedFormatError,
)

logger = get_logger(__name__)


def _error_payload(
    *,
    error_code: str,
    message: str,
    details: dict | None = None,
) -> dict:
    return {
        "status": "error",
        "error_code": error_code,
        "message": message,
        "details": details,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MediaFetchError)
    async def media_fetch_handler(
        _request: Request, exc: MediaFetchError
    ) -> JSONResponse:
        logger.warning("Music overlay media fetch failed: %s", exc.message)
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="media_fetch_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(MediaTooLargeError)
    async def media_too_large_handler(
        _request: Request, exc: MediaTooLargeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                error_code="media_too_large",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProbeError)
    async def probe_handler(_request: Request, exc: ProbeError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="media_probe_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(InvalidSettingsError)
    async def invalid_settings_handler(
        _request: Request, exc: InvalidSettingsError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="invalid_overlay_settings",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(UnsupportedFormatError)
    async def unsupported_format_handler(
        _request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="unsupported_format",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(OverlayError)
    async def overlay_handler(_request: Request, exc: OverlayError) -> JSONResponse:
        logger.error("Music overlay ffmpeg error: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="overlay_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(JobNotFoundError)
    async def not_found_handler(
        _request: Request, exc: JobNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="job_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(MusicOverlayError)
    async def music_overlay_handler(
        _request: Request, exc: MusicOverlayError
    ) -> JSONResponse:
        logger.error("Music overlay error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="music_overlay_error",
                message=exc.message,
                details=exc.details,
            ),
        )
