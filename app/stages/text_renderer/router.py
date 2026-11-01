from uuid import UUID

from fastapi import APIRouter, Body, Depends

from app.stages.text_renderer.deps import get_text_renderer_service
from app.stages.text_renderer.schemas import (
    ErrorResponse,
    StoredRenderRequest,
    TextPreviewRequest,
    TextPreviewResponse,
    TextRenderRequest,
    TextRenderResponse,
    TextRerenderRequest,
)
from app.stages.text_renderer.service import TextRendererService

router = APIRouter(prefix="/v1/text-renderer", tags=["Text Renderer"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/render",
    response_model=TextRenderResponse,
    responses=_ERROR_RESPONSES,
    summary="Burn exact user text onto a generated video (or export without overlay)",
)
async def render_text(
    request: TextRenderRequest,
    service: TextRendererService = Depends(get_text_renderer_service),
) -> TextRenderResponse:
    """Deterministic FFmpeg/ASS burn-in. Source video is archived and never overwritten.

    Set `skip_text: true` to export a copy without overlay. Pass `segments` for
    timed captions, or `text` for a full-clip caption. Wording is never rewritten.
    """
    return await service.render(request)


@router.post(
    "/preview",
    response_model=TextPreviewResponse,
    responses=_ERROR_RESPONSES,
    summary="Render a still preview frame with the same text style and timing",
)
async def preview_text(
    request: TextPreviewRequest,
    service: TextRendererService = Depends(get_text_renderer_service),
) -> TextPreviewResponse:
    """Useful for UI iteration before committing a full video burn-in."""
    return await service.preview(request)


@router.get(
    "/renders/{render_id}",
    response_model=TextRenderResponse,
    responses=_ERROR_RESPONSES,
    summary="Fetch a previous text render",
)
async def get_text_render(
    render_id: UUID,
    service: TextRendererService = Depends(get_text_renderer_service),
) -> TextRenderResponse:
    return await service.get_render(render_id)


@router.get(
    "/renders/{render_id}/request",
    response_model=StoredRenderRequest,
    responses=_ERROR_RESPONSES,
    summary="Read archived source reference and text parameters",
)
async def get_text_render_request(
    render_id: UUID,
    service: TextRendererService = Depends(get_text_renderer_service),
) -> StoredRenderRequest:
    return await service.get_request(render_id)


@router.post(
    "/renders/{render_id}/rerender",
    response_model=TextRenderResponse,
    responses=_ERROR_RESPONSES,
    summary="Re-burn text onto the archived source without regenerating the video",
)
async def rerender_text(
    render_id: UUID,
    body: TextRerenderRequest = Body(default_factory=TextRerenderRequest),
    service: TextRendererService = Depends(get_text_renderer_service),
) -> TextRenderResponse:
    """Reuse the archived generated video. Optionally override text, style, or segments."""
    return await service.rerender(render_id, body)
