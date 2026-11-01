from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger, setup_logging
from app.pipeline.handlers import register_exception_handlers as register_pipeline_handlers
from app.stages.face_replacement.handlers import (
    register_exception_handlers as register_face_replacement_handlers,
)
from app.stages.hook_video_generator.handlers import (
    register_exception_handlers as register_hook_video_handlers,
)
from app.stages.hook_analyzer.handlers import register_exception_handlers as register_hook_handlers
from app.stages.hook_text_engine.handlers import (
    register_exception_handlers as register_hook_text_handlers,
)
from app.stages.music_overlay.handlers import (
    register_exception_handlers as register_music_overlay_handlers,
)
from app.stages.product_demo_composer.handlers import (
    register_exception_handlers as register_product_demo_composer_handlers,
)
from app.stages.text_renderer.handlers import (
    register_exception_handlers as register_text_renderer_handlers,
)
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
            "AI UGC Pipeline. Independent stage APIs under `/v1/...`, "
            "session orchestration under `/v1/pipeline`, reusable uploads under `/v1/assets`. "
            "Local demo UI: `/demo`."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.include_router(api_router)

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @application.get("/demo", include_in_schema=False)
        async def demo_page() -> FileResponse:
            return FileResponse(static_dir / "demo.html")

    register_ingestion_handlers(application)
    register_hook_handlers(application)
    register_hook_text_handlers(application)
    register_face_replacement_handlers(application)
    register_hook_video_handlers(application)
    register_text_renderer_handlers(application)
    register_product_demo_composer_handlers(application)
    register_music_overlay_handlers(application)
    register_pipeline_handlers(application)
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
                details={"errors": jsonable_encoder(exc.errors())},
            ),
        )


app = create_app()
