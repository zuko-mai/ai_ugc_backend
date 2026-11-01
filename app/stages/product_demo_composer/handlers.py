from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.product_demo_composer.exceptions import (
    ComposeError,
    InvalidSettingsError,
    JobNotFoundError,
    ProbeError,
    ProductDemoComposerError,
    UnsupportedFormatError,
    VideoFetchError,
    VideoTooLargeError,
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
    @app.exception_handler(VideoFetchError)
    async def video_fetch_handler(
        _request: Request, exc: VideoFetchError
    ) -> JSONResponse:
        logger.warning("Product demo composer video fetch failed: %s", exc.message)
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="video_fetch_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VideoTooLargeError)
    async def video_too_large_handler(
        _request: Request, exc: VideoTooLargeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                error_code="video_too_large",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProbeError)
    async def probe_handler(_request: Request, exc: ProbeError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="video_probe_failed",
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
                error_code="invalid_compose_settings",
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

    @app.exception_handler(ComposeError)
    async def compose_handler(_request: Request, exc: ComposeError) -> JSONResponse:
        logger.error("Product demo composer ffmpeg error: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="compose_failed",
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

    @app.exception_handler(ProductDemoComposerError)
    async def composer_handler(
        _request: Request, exc: ProductDemoComposerError
    ) -> JSONResponse:
        logger.error("Product demo composer error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="product_demo_composer_error",
                message=exc.message,
                details=exc.details,
            ),
        )
