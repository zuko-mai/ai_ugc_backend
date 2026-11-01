from uuid import UUID

from fastapi import APIRouter, Depends

from app.stages.hook_text_engine.deps import get_hook_text_engine_service
from app.stages.hook_text_engine.schemas import (
    AdaptationRequest,
    AdaptationResponse,
    ErrorResponse,
    ExtractionRequest,
    ExtractionResponse,
    SelectRequest,
    SelectResponse,
    SessionResponse,
)
from app.stages.hook_text_engine.service import HookTextEngineService

router = APIRouter(prefix="/v1/hook-text-engine", tags=["Hook Text Engine"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
}


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses=_ERROR_RESPONSES,
    summary="Extract on-screen (OCR) and spoken (ASR) hook text separately",
)
async def extract(
    body: ExtractionRequest,
    service: HookTextEngineService = Depends(get_hook_text_engine_service),
) -> ExtractionResponse:
    return await service.extract(body)


@router.post(
    "/adapt",
    response_model=AdaptationResponse,
    responses=_ERROR_RESPONSES,
    summary="Adapt hook text for a product (or skip and keep original)",
)
async def adapt(
    body: AdaptationRequest,
    service: HookTextEngineService = Depends(get_hook_text_engine_service),
) -> AdaptationResponse:
    return await service.adapt(body)


@router.post(
    "/select",
    response_model=SelectResponse,
    responses=_ERROR_RESPONSES,
    summary="Select final hook text (manual > variant > original)",
)
async def select(
    body: SelectRequest,
    service: HookTextEngineService = Depends(get_hook_text_engine_service),
) -> SelectResponse:
    return await service.select(body)


@router.get(
    "/sessions/{adaptation_id}",
    response_model=SessionResponse,
    responses=_ERROR_RESPONSES,
    summary="Load stored original, variants, and selected text",
)
async def get_session(
    adaptation_id: UUID,
    service: HookTextEngineService = Depends(get_hook_text_engine_service),
) -> SessionResponse:
    return await service.get_session(adaptation_id)
