from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.text_renderer.service import TextRendererService


def get_text_renderer_service() -> TextRendererService:
    return TextRendererService(storage=get_storage(), settings=get_settings())
