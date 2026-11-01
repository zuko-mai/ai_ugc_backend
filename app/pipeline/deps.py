from app.api.deps import get_storage
from app.core.config import get_settings
from app.pipeline.service import PipelineService
from app.pipeline.store import AssetStore


def get_pipeline_service() -> PipelineService:
    return PipelineService(get_storage(), get_settings())


def get_asset_store() -> AssetStore:
    return AssetStore(get_storage())
