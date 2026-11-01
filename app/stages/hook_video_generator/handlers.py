from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.hook_video_generator.exceptions import (
    ComparisonNotFoundError,
    FrameDecodeError,
    FrameFetchError,
    FrameTooLargeError,
    GenerationFailedError,
    HookVideoError,
    JobNotFoundError,
    ProviderApiError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
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
    @app.exception_handler(FrameFetchError)
    async def frame_fetch_handler(_request: Request, exc: FrameFetchError) -> JSONResponse:
        logger.warning("Hook video frame fetch failed: %s", exc.message)
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="frame_fetch_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(FrameTooLargeError)
    async def frame_too_large_handler(
        _request: Request, exc: FrameTooLargeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                error_code="frame_too_large",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(FrameDecodeError)
    async def frame_decode_handler(_request: Request, exc: FrameDecodeError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="frame_decode_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProviderNotConfiguredError)
    async def provider_config_handler(
        _request: Request, exc: ProviderNotConfiguredError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=_error_payload(
                error_code="provider_not_configured",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_handler(
        _request: Request, exc: ProviderTimeoutError
    ) -> JSONResponse:
        logger.error("Hook video provider timeout: %s", exc.message)
        return JSONResponse(
            status_code=504,
            content=_error_payload(
                error_code="provider_timeout",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProviderApiError)
    async def provider_api_handler(_request: Request, exc: ProviderApiError) -> JSONResponse:
        logger.error("Hook video provider API error: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="provider_api_error",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(GenerationFailedError)
    async def generation_failed_handler(
        _request: Request, exc: GenerationFailedError
    ) -> JSONResponse:
        logger.error("Hook video generation failed: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="generation_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(_request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="job_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ComparisonNotFoundError)
    async def comparison_not_found_handler(
        _request: Request, exc: ComparisonNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="comparison_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(HookVideoError)
    async def hook_video_handler(_request: Request, exc: HookVideoError) -> JSONResponse:
        logger.error("Hook video error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="hook_video_error",
                message=exc.message,
                details=exc.details,
            ),
        )
