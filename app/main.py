from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger, setup_logging
from app.stages.video_ingestion.handlers import register_exception_handlers as register_ingestion_handlers

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting %s v%s env=%s storage=%s",
        settings.app_name,
        __version__,
        settings.app_env,
        settings.storage_root,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="AI UGC Creator",
        description=(
            "AI UGC Pipeline. Stages are independent API groups under `/v1/...` "
            "and will later be composed into a single pipeline."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.include_router(api_router)

    register_ingestion_handlers(application)
    _register_shared_handlers(application)
    return application


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


def _register_shared_handlers(application: FastAPI) -> None:
    @application.exception_handler(StorageError)
    async def storage_handler(_request: Request, exc: StorageError) -> JSONResponse:
        logger.error("Storage error: %s details=%s", exc.message, exc.details)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                error_code="storage_error",
                message=exc.message,
                details=exc.details,
            ),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error_code="validation_error",
                message="Request validation failed",
                details={"errors": exc.errors()},
            ),
        )


app = create_app()
