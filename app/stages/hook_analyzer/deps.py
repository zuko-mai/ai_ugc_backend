from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.hook_analyzer.service import HookAnalyzerService


def get_hook_analyzer_service() -> HookAnalyzerService:
    return HookAnalyzerService(storage=get_storage(), settings=get_settings())
