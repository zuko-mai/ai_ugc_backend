from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.product_demo_composer.service import ProductDemoComposerService


def get_product_demo_composer_service() -> ProductDemoComposerService:
    return ProductDemoComposerService(storage=get_storage(), settings=get_settings())
