from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.hook_analyzer.exceptions import (
    FrameExtractionError,
    FrameNotFoundError,
    GeminiMotionPromptError,
    HookAnalyzerError,
    InvalidIntervalError,
    SessionNotFoundError,
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
        logger.warning("Hook video not found: %s", exc.message)
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

    @app.exception_handler(FrameNotFoundError)
    async def frame_not_found_handler(
        _request: Request, exc: FrameNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="frame_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(FrameExtractionError)
    async def frame_extraction_handler(
        _request: Request, exc: FrameExtractionError
    ) -> JSONResponse:
        logger.error("Frame extraction failed: %s", exc.message)
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="frame_extraction_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(GeminiMotionPromptError)
    async def gemini_handler(
        _request: Request, exc: GeminiMotionPromptError
    ) -> JSONResponse:
        logger.error("Gemini motionPrompt failed: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="gemini_motion_prompt_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(HookAnalyzerError)
    async def hook_analyzer_handler(
        _request: Request, exc: HookAnalyzerError
    ) -> JSONResponse:
        logger.error("Hook analyzer error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="hook_analyzer_error",
                message=exc.message,
                details=exc.details,
            ),
        )
