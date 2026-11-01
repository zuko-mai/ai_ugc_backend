from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.hook_text_engine.exceptions import (
    AdaptationError,
    ExtractionError,
    HookTextEngineError,
    InvalidIntervalError,
    SessionNotFoundError,
    VariantNotFoundError,
    VideoNotFoundError,
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
    @app.exception_handler(InvalidIntervalError)
    async def invalid_interval_handler(
        _request: Request, exc: InvalidIntervalError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="invalid_interval",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VideoNotFoundError)
    async def video_not_found_handler(
        _request: Request, exc: VideoNotFoundError
    ) -> JSONResponse:
        logger.warning("Hook text video not found: %s", exc.message)
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="video_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        _request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="session_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(VariantNotFoundError)
    async def variant_not_found_handler(
        _request: Request, exc: VariantNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="variant_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ExtractionError)
    async def extraction_handler(
        _request: Request, exc: ExtractionError
    ) -> JSONResponse:
        logger.error("Hook text extraction failed: %s", exc.message)
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="extraction_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(AdaptationError)
    async def adaptation_handler(
        _request: Request, exc: AdaptationError
    ) -> JSONResponse:
        logger.error("Hook text adaptation failed: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="adaptation_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(HookTextEngineError)
    async def hook_text_handler(
        _request: Request, exc: HookTextEngineError
    ) -> JSONResponse:
        logger.error("Hook text engine error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="hook_text_engine_error",
                message=exc.message,
                details=exc.details,
            ),
        )
