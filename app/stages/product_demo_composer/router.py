from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.stages.product_demo_composer.deps import get_product_demo_composer_service
from app.stages.product_demo_composer.exceptions import InvalidSettingsError
from app.stages.product_demo_composer.schemas import (
    ComposeAccepted,
    ComposeJobView,
    ComposeRequest,
    ComposeRerunRequest,
    ComposeSettings,
    ErrorResponse,
    StoredComposeRequest,
    VideoAssetRef,
)
from app.stages.product_demo_composer.service import ProductDemoComposerService

router = APIRouter(prefix="/v1/product-demo-composer", tags=["Product Demo Composer"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
}


@router.post(
    "/compose",
    response_model=ComposeAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue composition of AI Hook followed by Product Demo",
)
async def compose_product_demo(
    project_id: UUID = Form(
        ...,
        description="Pipeline project id.",
    ),
    hook_asset_id: UUID = Form(
        ...,
        description="Asset id of the AI Hook video.",
    ),
    hook_video_url: str = Form(
        ...,
        min_length=1,
        description="URL of the AI Hook video (/media/... or http(s)).",
    ),
    demo_video: UploadFile = File(
        ...,
        description="Product demo video file (MP4 or MOV). Uploaded in Swagger as a file.",
    ),
    demo_asset_id: UUID | None = Form(
        None,
        description="Optional asset id for the uploaded demo. Auto-generated if omitted.",
    ),
    output_format: str = Form("mp4", description="mp4 or mov"),
    aspect_ratio: str = Form("9:16", description="9:16, 16:9, or 1:1"),
    resolution: str = Form("1080x1920", description="IG/TikTok default: 1080x1920"),
    fps: int = Form(30, ge=12, le=60),
    fit_mode: str = Form(
        "blur",
        description="blur (default — blurred fill) | cover (crop) | contain (black bars)",
    ),
    transition: str = Form("none", description="none | fade | crossfade"),
    transition_duration_seconds: float = Form(0, ge=0, le=5),
    audio_mode: str = Form(
        "both",
        description="both | hook | demo | none",
    ),
    service: ProductDemoComposerService = Depends(get_product_demo_composer_service),
) -> ComposeAccepted:
    """Join two videos as [AI HOOK] → [PRODUCT DEMO].

    **Content-Type:** `multipart/form-data`

    - Hook: URL from a previous stage
    - Demo: uploaded file (visible as a file picker in Swagger)
    - Sources are archived unchanged; poll `GET /jobs/{job_id}` for the result
    """
    try:
        settings = ComposeSettings(
            output_format=output_format,  # type: ignore[arg-type]
            aspect_ratio=aspect_ratio,  # type: ignore[arg-type]
            resolution=resolution,
            fps=fps,
            fit_mode=fit_mode,  # type: ignore[arg-type]
            transition=transition,  # type: ignore[arg-type]
            transition_duration_seconds=transition_duration_seconds,
            audio_mode=audio_mode,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise InvalidSettingsError(
            "Invalid compose settings",
            details={"errors": jsonable_encoder(exc.errors())},
        ) from exc
    return await service.submit(
        project_id=project_id,
        hook_video=VideoAssetRef(asset_id=hook_asset_id, video_url=hook_video_url),
        demo_video=demo_video,
        demo_asset_id=demo_asset_id,
        settings=settings,
    )


@router.post(
    "/compose/refs",
    response_model=ComposeAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Compose from stored hook + demo URLs (no file re-upload)",
)
async def compose_product_demo_refs(
    body: ComposeRequest,
    service: ProductDemoComposerService = Depends(get_product_demo_composer_service),
) -> ComposeAccepted:
    """Reuse previously uploaded/stored videos by asset_id + video_url."""
    return await service.submit_refs(
        project_id=body.project_id,
        hook_video=body.hook_video,
        demo_video=body.demo_video,
        settings=body.settings,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ComposeJobView,
    responses=_ERROR_RESPONSES,
    summary="Poll a product demo composition job",
)
async def get_compose_job(
    job_id: UUID,
    service: ProductDemoComposerService = Depends(get_product_demo_composer_service),
) -> ComposeJobView:
    """On success returns asset_id, video_url, duration, geometry, and segments."""
    return await service.get_job(job_id)


@router.get(
    "/jobs/{job_id}/request",
    response_model=StoredComposeRequest,
    responses=_ERROR_RESPONSES,
    summary="Read archived input references and compose settings",
)
async def get_compose_request(
    job_id: UUID,
    service: ProductDemoComposerService = Depends(get_product_demo_composer_service),
) -> StoredComposeRequest:
    return await service.get_request(job_id)


@router.post(
    "/jobs/{job_id}/rerun",
    response_model=ComposeAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Recompose from archived inputs with optional new settings",
)
async def rerun_compose(
    job_id: UUID,
    body: ComposeRerunRequest = Body(default_factory=ComposeRerunRequest),
    service: ProductDemoComposerService = Depends(get_product_demo_composer_service),
) -> ComposeAccepted:
    """Reuse archived hook/demo originals. Optionally override transition/audio/format."""
    return await service.rerun(job_id, body)
