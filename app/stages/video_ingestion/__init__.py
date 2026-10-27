"""Video Ingestion stage — validate, store original, extract metadata."""

from app.stages.video_ingestion.router import router

__all__ = ["router"]
