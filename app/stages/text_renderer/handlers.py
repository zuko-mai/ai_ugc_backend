from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.text_renderer.exceptions import (
    FontNotFoundError,
    InvalidTextError,
    ProbeError,
    RenderError,
    RenderNotFoundError,
    TextRendererError,
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
        logger.warning("Text renderer video fetch failed: %s", exc.message)
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

    @app.exception_handler(FontNotFoundError)
    async def font_handler(_request: Request, exc: FontNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=_error_payload(
                error_code="font_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(InvalidTextError)
    async def invalid_text_handler(
        _request: Request, exc: InvalidTextError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="invalid_text",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(RenderError)
    async def render_handler(_request: Request, exc: RenderError) -> JSONResponse:
        logger.error("Text renderer ffmpeg error: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="render_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(RenderNotFoundError)
    async def not_found_handler(
        _request: Request, exc: RenderNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="render_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(TextRendererError)
    async def text_renderer_handler(
        _request: Request, exc: TextRendererError
    ) -> JSONResponse:
        logger.error("Text renderer error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="text_renderer_error",
                message=exc.message,
                details=exc.details,
            ),
        )
