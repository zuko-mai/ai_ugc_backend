from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_storage
from app.core.exceptions import StorageError
from app.storage.base import FileStorage

router = APIRouter(tags=["Media"])


@router.get("/media/{object_path:path}")
async def serve_media(
    object_path: str,
    storage: FileStorage = Depends(get_storage),
) -> FileResponse:
    """Serve stored originals for downstream stages (local storage)."""
    try:
        path: Path = storage.get_local_path(object_path)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc

    suffix = path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)
