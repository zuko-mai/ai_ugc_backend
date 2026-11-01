from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.hook_video_generator.service import HookVideoService


def get_hook_video_service() -> HookVideoService:
    return HookVideoService(storage=get_storage(), settings=get_settings())
