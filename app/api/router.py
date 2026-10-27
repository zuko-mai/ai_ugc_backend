from fastapi import APIRouter

from app.api.routes import health, media
from app.stages.video_ingestion import router as video_ingestion_router

api_router = APIRouter()

# Shared infrastructure routes
api_router.include_router(health.router)
api_router.include_router(media.router)

# Independent pipeline stages (add new stages here as separate routers)
api_router.include_router(video_ingestion_router)
