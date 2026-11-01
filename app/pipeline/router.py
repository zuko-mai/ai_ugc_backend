from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile

from app.pipeline.deps import get_asset_store, get_pipeline_service
from app.pipeline.schemas import (
    AssetKind,
    AssetResponse,
    CandidateOptions,
    HookTextOptions,
    HookVideoOptions,
    IngestUrlRequest,
    PipelineSession,
    Product,
    RenderOptions,
    SelectFrameRequest,
    SelectTextRequest,
)
from app.pipeline.service import PipelineService
from app.pipeline.store import AssetStore

assets_router = APIRouter(prefix="/v1/assets", tags=["Pipeline Assets"])
router = APIRouter(prefix="/v1/pipeline", tags=["Pipeline"])


@assets_router.post("/upload", response_model=AssetResponse)
async def upload_asset(
    file: UploadFile = File(...),
    kind: AssetKind = Form("other"),
    store: AssetStore = Depends(get_asset_store),
):
    return await store.upload(file, kind)


@assets_router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: UUID,
    store: AssetStore = Depends(get_asset_store),
):
    return await store.get(asset_id)


@router.post("/sessions", response_model=PipelineSession, status_code=201)
async def create_session(
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.create_session()


@router.get("/sessions/{session_id}", response_model=PipelineSession)
async def get_session(
    session_id: UUID,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.get_session(session_id)


@router.put("/sessions/{session_id}/product", response_model=PipelineSession)
async def set_product(
    session_id: UUID,
    body: Product,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.set_product(session_id, body)


@router.post("/sessions/{session_id}/ingest/upload", response_model=PipelineSession)
async def ingest_upload(
    session_id: UUID,
    file: UploadFile = File(...),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.ingest_upload(session_id, file)


@router.post("/sessions/{session_id}/ingest/url", response_model=PipelineSession)
async def ingest_url(
    session_id: UUID,
    body: IngestUrlRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.ingest_url(session_id, body.url)


@router.post("/sessions/{session_id}/hook/candidates", response_model=PipelineSession)
async def find_candidates(
    session_id: UUID,
    body: CandidateOptions = Body(default_factory=CandidateOptions),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.find_candidates(session_id, body)


@router.post("/sessions/{session_id}/hook/select", response_model=PipelineSession)
async def select_frame(
    session_id: UUID,
    body: SelectFrameRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.select_frame(session_id, body)


@router.post("/sessions/{session_id}/face/generate", response_model=PipelineSession)
async def generate_face(
    session_id: UUID,
    face_file: UploadFile | None = File(None),
    face_asset_id: UUID | None = Form(None),
    user_prompt: str | None = Form(None),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.generate_face(
        session_id,
        face_file=face_file,
        face_asset_id=face_asset_id,
        user_prompt=user_prompt,
    )


@router.post(
    "/sessions/{session_id}/hook-video/generate",
    response_model=PipelineSession,
    status_code=202,
)
async def generate_hook_video(
    session_id: UUID,
    body: HookVideoOptions = Body(default_factory=HookVideoOptions),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.generate_hook_video(session_id, body)


@router.post("/sessions/{session_id}/hook-video/refresh", response_model=PipelineSession)
async def refresh_hook_video(
    session_id: UUID,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.refresh_hook_video(session_id)


@router.post(
    "/sessions/{session_id}/hook-text/extract-adapt",
    response_model=PipelineSession,
)
async def extract_adapt_text(
    session_id: UUID,
    body: HookTextOptions = Body(default_factory=HookTextOptions),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.extract_adapt_text(session_id, body)


@router.post("/sessions/{session_id}/hook-text/select", response_model=PipelineSession)
async def select_text(
    session_id: UUID,
    body: SelectTextRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.select_text(session_id, body)


@router.post("/sessions/{session_id}/render", response_model=PipelineSession)
async def render_text(
    session_id: UUID,
    body: RenderOptions = Body(default_factory=RenderOptions),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.render_text(session_id, body)


@router.post(
    "/sessions/{session_id}/compose",
    response_model=PipelineSession,
    status_code=202,
)
async def compose(
    session_id: UUID,
    demo_file: UploadFile | None = File(None),
    demo_asset_id: UUID | None = Form(None),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.compose(
        session_id,
        demo_file=demo_file,
        demo_asset_id=demo_asset_id,
    )


@router.post("/sessions/{session_id}/compose/refresh", response_model=PipelineSession)
async def refresh_compose(
    session_id: UUID,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.refresh_composition(session_id)


@router.post(
    "/sessions/{session_id}/music",
    response_model=PipelineSession,
    status_code=202,
)
async def add_music(
    session_id: UUID,
    music_file: UploadFile | None = File(None),
    music_asset_id: UUID | None = Form(None),
    music_start_offset_seconds: float = Form(
        0,
        ge=0,
        description="Seek inside the music file (where the track should start playing from).",
    ),
    video_start_seconds: float = Form(
        0,
        ge=0,
        description="When on the video timeline the music becomes audible.",
    ),
    music_volume: float = Form(0.3, ge=0, le=1),
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.add_music(
        session_id,
        music_file=music_file,
        music_asset_id=music_asset_id,
        music_start_offset_seconds=music_start_offset_seconds,
        video_start_seconds=video_start_seconds,
        music_volume=music_volume,
    )


@router.post("/sessions/{session_id}/music/refresh", response_model=PipelineSession)
async def refresh_music(
    session_id: UUID,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineSession:
    return await service.refresh_music(session_id)
