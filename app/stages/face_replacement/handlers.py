from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.stages.face_replacement.exceptions import (
    FaceReplacementError,
    FrameNotSuitableError,
    GenerationFailedError,
    GenerationNotFoundError,
    ImageDecodeError,
    ImageFetchError,
    ImageResolutionError,
    ImageTooLargeError,
    MissingFaceReferenceError,
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
    @app.exception_handler(ImageFetchError)
    async def image_fetch_handler(
        _request: Request, exc: ImageFetchError
    ) -> JSONResponse:
        logger.warning("Face replacement image fetch failed: %s", exc.message)
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="image_fetch_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ImageTooLargeError)
    async def image_too_large_handler(
        _request: Request, exc: ImageTooLargeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content=_error_payload(
                error_code="image_too_large",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ImageResolutionError)
    async def image_resolution_handler(
        _request: Request, exc: ImageResolutionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="image_resolution_too_low",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ImageDecodeError)
    async def image_decode_handler(
        _request: Request, exc: ImageDecodeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="image_decode_failed",
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
        logger.error("Face replacement provider timeout: %s", exc.message)
        return JSONResponse(
            status_code=504,
            content=_error_payload(
                error_code="provider_timeout",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(ProviderApiError)
    async def provider_api_handler(
        _request: Request, exc: ProviderApiError
    ) -> JSONResponse:
        logger.error("Face replacement provider API error: %s", exc.message)
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
        logger.error("Face replacement generation failed: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content=_error_payload(
                error_code="generation_failed",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(FrameNotSuitableError)
    async def frame_not_suitable_handler(
        _request: Request, exc: FrameNotSuitableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="frame_not_suitable",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(GenerationNotFoundError)
    async def generation_not_found_handler(
        _request: Request, exc: GenerationNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_payload(
                error_code="generation_not_found",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(MissingFaceReferenceError)
    async def missing_face_handler(
        _request: Request, exc: MissingFaceReferenceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="missing_face_reference",
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(FaceReplacementError)
    async def face_replacement_handler(
        _request: Request, exc: FaceReplacementError
    ) -> JSONResponse:
        logger.error("Face replacement error: %s", exc.message)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                error_code="face_replacement_error",
                message=exc.message,
                details=exc.details,
            ),
        )
