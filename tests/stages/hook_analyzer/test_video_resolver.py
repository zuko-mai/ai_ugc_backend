from uuid import uuid4

import pytest

from app.core.config import Settings
from app.stages.hook_analyzer.exceptions import VideoNotFoundError
from app.stages.hook_analyzer.video_resolver import resolve_video_ref
from app.storage.local import LocalFileStorage


@pytest.mark.asyncio
async def test_resolve_by_video_id(
    settings: Settings, storage: LocalFileStorage, sample_mp4
) -> None:
    video_id = uuid4()
    key = f"originals/{video_id}/source.mp4"
    await storage.save_file(key, sample_mp4, content_type="video/mp4")

    resolved_id, path = await resolve_video_ref(
        video_id=video_id,
        video_url=None,
        storage=storage,
        settings=settings,
    )
    assert resolved_id == video_id
    assert path.is_file()


@pytest.mark.asyncio
async def test_resolve_missing_id_raises(settings: Settings, storage: LocalFileStorage) -> None:
    with pytest.raises(VideoNotFoundError):
        await resolve_video_ref(
            video_id=uuid4(),
            video_url=None,
            storage=storage,
            settings=settings,
        )
