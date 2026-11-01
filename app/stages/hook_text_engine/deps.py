from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.hook_text_engine.service import HookTextEngineService


def get_hook_text_engine_service() -> HookTextEngineService:
    return HookTextEngineService(storage=get_storage(), settings=get_settings())
