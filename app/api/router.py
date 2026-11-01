from fastapi import APIRouter

from app.api.routes import health, media
from app.pipeline import assets_router, router as pipeline_router
from app.stages.face_replacement import router as face_replacement_router
from app.stages.hook_analyzer import router as hook_analyzer_router
from app.stages.hook_text_engine import router as hook_text_engine_router
from app.stages.hook_video_generator import router as hook_video_router
from app.stages.music_overlay import router as music_overlay_router
from app.stages.product_demo_composer import router as product_demo_composer_router
from app.stages.text_renderer import router as text_renderer_router
from app.stages.video_ingestion import router as video_ingestion_router

api_router = APIRouter()

# Shared infrastructure routes
api_router.include_router(health.router)
api_router.include_router(media.router)
api_router.include_router(assets_router)
api_router.include_router(pipeline_router)

# Independent pipeline stages (add new stages here as separate routers)
api_router.include_router(video_ingestion_router)
api_router.include_router(hook_analyzer_router)
api_router.include_router(hook_text_engine_router)
api_router.include_router(face_replacement_router)
api_router.include_router(hook_video_router)
api_router.include_router(text_renderer_router)
api_router.include_router(product_demo_composer_router)
api_router.include_router(music_overlay_router)
