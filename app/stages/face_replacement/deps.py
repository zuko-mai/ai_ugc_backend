from app.api.deps import get_storage
from app.core.config import get_settings
from app.stages.face_replacement.service import FaceReplacementService


def get_face_replacement_service() -> FaceReplacementService:
    return FaceReplacementService(storage=get_storage(), settings=get_settings())
