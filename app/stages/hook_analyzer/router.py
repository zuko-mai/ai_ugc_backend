from fastapi import APIRouter, Depends

from app.stages.hook_analyzer.deps import get_hook_analyzer_service
from app.stages.hook_analyzer.schemas import (
    CandidatesRequest,
    CandidatesResponse,
    ErrorResponse,
    MotionPlanRequest,
    MotionPlanResponse,
    ReanalyzeRequest,
)
from app.stages.hook_analyzer.service import HookAnalyzerService

router = APIRouter(prefix="/v1/hook-analyzer", tags=["Hook Analyzer"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
}


@router.post(
    "/candidates",
    response_model=CandidatesResponse,
    responses=_ERROR_RESPONSES,
    summary="Find first-frame hook candidates in an interval",
)
async def find_candidates(
    body: CandidatesRequest,
    service: HookAnalyzerService = Depends(get_hook_analyzer_service),
) -> CandidatesResponse:
    return await service.find_candidates(body)


@router.post(
    "/motion-plan",
    response_model=MotionPlanResponse,
    responses=_ERROR_RESPONSES,
    summary="Build motion/emotion plan for the hook interval",
)
async def motion_plan(
    body: MotionPlanRequest,
    service: HookAnalyzerService = Depends(get_hook_analyzer_service),
) -> MotionPlanResponse:
    return await service.build_motion_plan(body)


@router.post(
    "/reanalyze",
    response_model=MotionPlanResponse,
    responses=_ERROR_RESPONSES,
    summary="Reanalyze motion plan after user selects a candidate frame (uses cache)",
)
async def reanalyze(
    body: ReanalyzeRequest,
    service: HookAnalyzerService = Depends(get_hook_analyzer_service),
) -> MotionPlanResponse:
    return await service.reanalyze(body)
