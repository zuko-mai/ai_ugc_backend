from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.video_ingestion.service import VideoIngestionService


def get_ingestion_service() -> VideoIngestionService:
    return VideoIngestionService(storage=get_storage(), settings=get_settings())
