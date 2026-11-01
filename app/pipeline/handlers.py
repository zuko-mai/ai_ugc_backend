from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.pipeline.exceptions import (
    AssetNotFoundError,
    AssetUploadError,
    PipelineError,
    PipelineNotFoundError,
    PipelineOrderError,
)


def _payload(code: str, exc: PipelineError) -> dict:
    return {
        "status": "error",
        "error_code": code,
        "message": exc.message,
        "details": exc.details,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PipelineNotFoundError)
    async def session_not_found(
        _request: Request, exc: PipelineNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_payload("pipeline_session_not_found", exc),
        )

    @app.exception_handler(AssetNotFoundError)
    async def asset_not_found(
        _request: Request, exc: AssetNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content=_payload("asset_not_found", exc))

    @app.exception_handler(PipelineOrderError)
    async def invalid_order(
        _request: Request, exc: PipelineOrderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=_payload("pipeline_order_error", exc),
        )

    @app.exception_handler(AssetUploadError)
    async def invalid_asset(
        _request: Request, exc: AssetUploadError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_payload("asset_upload_error", exc),
        )

    @app.exception_handler(PipelineError)
    async def pipeline_error(
        _request: Request, exc: PipelineError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content=_payload("pipeline_error", exc))
