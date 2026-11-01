from uuid import UUID

from fastapi import APIRouter, Body, Depends

from app.stages.hook_video_generator.deps import get_hook_video_service
from app.stages.hook_video_generator.providers.registry import list_providers
from app.stages.hook_video_generator.schemas import (
    ErrorResponse,
    HookVideoAccepted,
    HookVideoCallback,
    HookVideoCompareAccepted,
    HookVideoCompareRequest,
    HookVideoComparisonView,
    HookVideoGenerateRequest,
    HookVideoJobView,
    HookVideoRerunRequest,
    ProviderListResponse,
    StoredGenerationRequest,
)
from app.stages.hook_video_generator.service import HookVideoService

router = APIRouter(prefix="/v1/hook-video", tags=["Hook Video"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


@router.get("/providers", response_model=ProviderListResponse)
async def list_hook_video_providers() -> ProviderListResponse:
    """Provider ids that can be passed as model_provider."""
    return ProviderListResponse(providers=list_providers())


@router.post(
    "/generate",
    response_model=HookVideoAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue an image-to-video hook from a first frame and a motion plan",
)
async def generate_hook_video(
    request: HookVideoGenerateRequest,
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoAccepted:
    """Queue generation. The response status is `queued` or `processing`.

    `success` and `generated_video_url` appear on `GET /jobs/{job_id}` only after
    the output file is stored and its duration, container, and resolution match
    the request. Poll that endpoint, or POST `/jobs/{job_id}/callback` when the
    vendor finishes.

    Change `model_provider` and `model_name` to switch models. The request body
    stays the same.
    """
    return await service.submit(request)


@router.post(
    "/compare",
    response_model=HookVideoCompareAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue the same hook on two or more models",
)
async def compare_hook_video_models(
    request: HookVideoCompareRequest,
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoCompareAccepted:
    """Submit one task per model. Poll `GET /comparisons/{comparison_id}`."""
    return await service.compare(request)


@router.get(
    "/jobs/{job_id}",
    response_model=HookVideoJobView,
    responses=_ERROR_RESPONSES,
    summary="Poll a hook video job",
)
async def get_hook_video_job(
    job_id: UUID,
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoJobView:
    """Advance the provider once when the job is still open."""
    return await service.get_job(job_id)


@router.get(
    "/jobs/{job_id}/request",
    response_model=StoredGenerationRequest,
    responses=_ERROR_RESPONSES,
    summary="Read the stored frame reference and generation parameters",
)
async def get_hook_video_request(
    job_id: UUID,
    service: HookVideoService = Depends(get_hook_video_service),
) -> StoredGenerationRequest:
    return await service.get_request(job_id)


@router.post(
    "/jobs/{job_id}/rerun",
    response_model=HookVideoAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue a new job from the stored first frame and parameters",
)
async def rerun_hook_video(
    job_id: UUID,
    body: HookVideoRerunRequest = Body(default_factory=HookVideoRerunRequest),
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoAccepted:
    """Reuse the archived frame and motion plan. Optionally override the model."""
    return await service.rerun(job_id, body)


@router.post(
    "/jobs/{job_id}/callback",
    response_model=HookVideoJobView,
    responses=_ERROR_RESPONSES,
    summary="Accept a provider completion callback",
)
async def hook_video_callback(
    job_id: UUID,
    body: HookVideoCallback,
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoJobView:
    """A succeeded callback is validated before the job can become `success`."""
    return await service.handle_callback(job_id, body)


@router.get(
    "/comparisons/{comparison_id}",
    response_model=HookVideoComparisonView,
    responses=_ERROR_RESPONSES,
    summary="Poll every job in a model comparison",
)
async def get_hook_video_comparison(
    comparison_id: UUID,
    service: HookVideoService = Depends(get_hook_video_service),
) -> HookVideoComparisonView:
    return await service.get_comparison(comparison_id)
