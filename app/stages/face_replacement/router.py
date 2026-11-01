from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.stages.face_replacement.deps import get_face_replacement_service
from app.stages.face_replacement.examples import (
    TEST_SELECTED_FRAME_URL,
    TEST_VIDEO_ID,
    TEST_VIDEO_URL,
)
from app.stages.face_replacement.schemas import (
    ErrorResponse,
    FaceReplacementRequest,
    FaceReplacementResponse,
    FaceRerunRequest,
    GenerationRecord,
)
from app.stages.face_replacement.service import FaceReplacementService

router = APIRouter(prefix="/v1/face-replacement", tags=["Face Replacement"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


@router.post(
    "/generate",
    response_model=FaceReplacementResponse,
    responses=_ERROR_RESPONSES,
    summary="Replace character on a selected UGC frame (first frame for i2v)",
)
async def generate_face_replacement(
    video_id: UUID = Form(
        ...,
        description="REQUIRED. Video id from ingestion / earlier stages.",
        examples=[TEST_VIDEO_ID],
    ),
    selected_frame_url: str = Form(
        ...,
        description="REQUIRED. URL of the selected UGC frame (master scene / FIRST image).",
        min_length=1,
        examples=[TEST_SELECTED_FRAME_URL],
    ),
    face_reference: UploadFile | None = File(
        None,
        description=(
            "Identity reference image (SECOND image). "
            "Required unless face_reference_url is provided."
        ),
    ),
    face_reference_url: str | None = Form(
        None,
        description="OPTIONAL. Reuse a previously stored face reference URL.",
    ),
    video_url: str | None = Form(
        None,
        description="OPTIONAL. Source video URL for pipeline/context.",
        examples=[TEST_VIDEO_URL],
    ),
    user_prompt: str | None = Form(
        None,
        description="OPTIONAL. Extra prompt instructions for regeneration/editing.",
    ),
    remove_overlay_text: bool = Form(
        True,
        description="OPTIONAL. Remove burned-in captions/stickers. Default: true.",
    ),
    preserve_original_composition: bool = Form(
        True,
        description="OPTIONAL. Keep framing/composition of the base frame. Default: true.",
    ),
    preserve_pose: bool = Form(
        True,
        description="OPTIONAL. Keep head/body pose from the base frame. Default: true.",
    ),
    preserve_lighting: bool = Form(
        True,
        description="OPTIONAL. Match lighting/shadows of the base frame. Default: true.",
    ),
    model_provider: str = Form(
        "gemini",
        description="OPTIONAL. Provider id (gemini, stub). Default: gemini.",
    ),
    model_name: str = Form(
        "gemini-3-pro-image",
        description=(
            "OPTIONAL. Default: gemini-3-pro-image (Nano Banana Pro). "
            "Secondary: gemini-3.1-flash-image (Nano Banana 2)."
        ),
        examples=["gemini-3-pro-image", "gemini-3.1-flash-image"],
    ),
    service: FaceReplacementService = Depends(get_face_replacement_service),
) -> FaceReplacementResponse:
    """Full character/identity replacement for the first UGC frame.

    Provide either multipart `face_reference` or `face_reference_url` to reuse
    a previously uploaded face without re-sending the file.
    """
    request = FaceReplacementRequest(
        video_id=video_id,
        video_url=video_url,
        selected_frame_url=selected_frame_url,
        face_reference_url=face_reference_url,
        user_prompt=user_prompt,
        remove_overlay_text=remove_overlay_text,
        preserve_original_composition=preserve_original_composition,
        preserve_pose=preserve_pose,
        preserve_lighting=preserve_lighting,
        model_provider=model_provider,
        model_name=model_name,
    )
    return await service.replace_face(request, face_reference=face_reference)


@router.get(
    "/generations/{generation_id}",
    response_model=FaceReplacementResponse,
    responses=_ERROR_RESPONSES,
    summary="Load a previous face-replacement generation",
)
async def get_face_generation(
    generation_id: UUID,
    service: FaceReplacementService = Depends(get_face_replacement_service),
) -> FaceReplacementResponse:
    return await service.get_generation(generation_id)


@router.get(
    "/generations/{generation_id}/request",
    response_model=GenerationRecord,
    responses=_ERROR_RESPONSES,
    summary="Read archived inputs and prompt for a generation",
)
async def get_face_generation_request(
    generation_id: UUID,
    service: FaceReplacementService = Depends(get_face_replacement_service),
) -> GenerationRecord:
    return await service.store.get_generation(generation_id)


@router.post(
    "/generations/{generation_id}/rerun",
    response_model=FaceReplacementResponse,
    responses=_ERROR_RESPONSES,
    summary="Regenerate using archived face + frame; optional prompt override",
)
async def rerun_face_generation(
    generation_id: UUID,
    body: FaceRerunRequest | None = None,
    service: FaceReplacementService = Depends(get_face_replacement_service),
) -> FaceReplacementResponse:
    return await service.rerun(generation_id, body)
