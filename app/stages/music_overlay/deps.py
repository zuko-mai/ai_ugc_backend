from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.music_overlay.service import MusicOverlayService


def get_music_overlay_service() -> MusicOverlayService:
    return MusicOverlayService(storage=get_storage(), settings=get_settings())
