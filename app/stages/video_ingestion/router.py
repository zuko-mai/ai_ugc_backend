from fastapi import APIRouter, Depends, File, UploadFile

from app.stages.video_ingestion.deps import get_ingestion_service
from app.stages.video_ingestion.schemas import (
    ErrorResponse,
    IngestFromUrlRequest,
    VideoIngestionResponse,
)
from app.stages.video_ingestion.service import VideoIngestionService

router = APIRouter(prefix="/v1/ingestion", tags=["Video Ingestion"])


@router.post(
    "/upload",
    response_model=VideoIngestionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Ingest video via multipart upload",
)
async def ingest_upload(
    file: UploadFile = File(..., description="MP4 or MOV video file"),
    service: VideoIngestionService = Depends(get_ingestion_service),
) -> VideoIngestionResponse:
    return await service.ingest_upload(file)


@router.post(
    "/url",
    response_model=VideoIngestionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Ingest video from a temporary URL",
)
async def ingest_from_url(
    body: IngestFromUrlRequest,
    service: VideoIngestionService = Depends(get_ingestion_service),
) -> VideoIngestionResponse:
    return await service.ingest_url(str(body.url))
