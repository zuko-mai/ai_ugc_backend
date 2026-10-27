from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.video_ingestion.exceptions import (
    MetadataExtractionError,
    UnsupportedFormatError,
    VideoIngestionError,
    VideoTooLargeError,
    VideoUnavailableError,
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
    """Stage-local error handlers — keep other stages independent."""

    @app.exception_handler(UnsupportedFormatError)
    async def unsupported_format_handler(
        _request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
        logger.warning("Unsupported format: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="unsupported_format",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VideoTooLargeError)
    async def too_large_handler(
        _request: Request, exc: VideoTooLargeError
    ) -> JSONResponse:
        logger.warning("Video too large: %s", exc.message)
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                error_code="video_too_large",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VideoUnavailableError)
    async def unavailable_handler(
        _request: Request, exc: VideoUnavailableError
    ) -> JSONResponse:
        logger.warning("Video unavailable: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="video_unavailable",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(MetadataExtractionError)
    async def metadata_handler(
        _request: Request, exc: MetadataExtractionError
    ) -> JSONResponse:
        logger.error("Metadata extraction failed: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="metadata_extraction_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VideoIngestionError)
    async def generic_ingestion_handler(
        _request: Request, exc: VideoIngestionError
    ) -> JSONResponse:
        logger.error("Ingestion error: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="ingestion_error",
                message=exc.message,
                details=exc.details,
            ),
        )
