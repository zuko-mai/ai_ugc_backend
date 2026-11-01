from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.stages.music_overlay.deps import get_music_overlay_service
from app.stages.music_overlay.exceptions import InvalidSettingsError
from app.stages.music_overlay.schemas import (
    ErrorResponse,
    OverlayAccepted,
    OverlayJobView,
    OverlayRequest,
    OverlayRerunRequest,
    OverlaySettings,
    StoredOverlayRequest,
    VideoAssetRef,
)
from app.stages.music_overlay.service import MusicOverlayService

router = APIRouter(prefix="/v1/music-overlay", tags=["Music Overlay"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
}


@router.post(
    "/overlay",
    response_model=OverlayAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue a music bed over a finished video",
)
async def overlay_music_json(
    body: OverlayRequest,
    service: MusicOverlayService = Depends(get_music_overlay_service),
) -> OverlayAccepted:
    """Mix `music.audio_url` onto `video.video_url`.

    `music_start_offset_seconds` seeks in the music file.
    `video_start_seconds` is when that excerpt starts on the video.
    Originals are archived unchanged. Poll `GET /jobs/{job_id}`.
    """
    return await service.submit(body)


@router.post(
    "/overlay/upload",
    response_model=OverlayAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Queue a mix using an uploaded music file",
)
async def overlay_music_upload(
    project_id: UUID = Form(...),
    video_asset_id: UUID = Form(...),
    video_url: str = Form(..., min_length=1),
    music: UploadFile = File(..., description="MP3, WAV, AAC, M4A, FLAC, OGG, or similar."),
    music_asset_id: UUID | None = Form(None),
    music_start_offset_seconds: float = Form(0, ge=0),
    video_start_seconds: float = Form(0, ge=0),
    music_volume: float = Form(0.3, ge=0, le=1),
    original_audio_enabled: bool = Form(True),
    original_audio_volume: float = Form(1.0, ge=0, le=1),
    fade_in_seconds: float = Form(0, ge=0, le=120),
    fade_out_seconds: float = Form(0, ge=0, le=120),
    loop_music: bool = Form(False),
    music_end_behavior: str = Form("silence"),
    output_format: str = Form("mp4"),
    service: MusicOverlayService = Depends(get_music_overlay_service),
) -> OverlayAccepted:
    """Upload music and reference the video by URL. Same job contract as `/overlay`."""
    try:
        settings = OverlaySettings(
            music_start_offset_seconds=music_start_offset_seconds,
            video_start_seconds=video_start_seconds,
            music_volume=music_volume,
            original_audio_enabled=original_audio_enabled,
            original_audio_volume=original_audio_volume,
            fade_in_seconds=fade_in_seconds,
            fade_out_seconds=fade_out_seconds,
            loop_music=loop_music,
            music_end_behavior=music_end_behavior,  # type: ignore[arg-type]
            output_format=output_format,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise InvalidSettingsError(
            "Invalid overlay settings",
            details={"errors": jsonable_encoder(exc.errors())},
        ) from exc
    return await service.submit_upload(
        project_id=project_id,
        video=VideoAssetRef(asset_id=video_asset_id, video_url=video_url),
        music=music,
        music_asset_id=music_asset_id,
        settings=settings,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=OverlayJobView,
    responses=_ERROR_RESPONSES,
    summary="Poll a music overlay job",
)
async def get_overlay_job(
    job_id: UUID,
    service: MusicOverlayService = Depends(get_music_overlay_service),
) -> OverlayJobView:
    """On success returns asset_id, video_url, duration, and the applied audio settings."""
    return await service.get_job(job_id)


@router.get(
    "/jobs/{job_id}/request",
    response_model=StoredOverlayRequest,
    responses=_ERROR_RESPONSES,
    summary="Read archived input references and overlay settings",
)
async def get_overlay_request(
    job_id: UUID,
    service: MusicOverlayService = Depends(get_music_overlay_service),
) -> StoredOverlayRequest:
    return await service.get_request(job_id)


@router.post(
    "/jobs/{job_id}/rerun",
    response_model=OverlayAccepted,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Mix again from archived inputs with optional new settings",
)
async def rerun_overlay(
    job_id: UUID,
    body: OverlayRerunRequest = Body(default_factory=OverlayRerunRequest),
    service: MusicOverlayService = Depends(get_music_overlay_service),
) -> OverlayAccepted:
    """Reuse archived video and music. Optionally override volume, offsets, or fades."""
    return await service.rerun(job_id, body)
