from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import StorageError
from app.storage.local import LocalFileStorage


async def _agen(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_save_stream_and_url(settings: Settings, storage: LocalFileStorage) -> None:
    key = await storage.save_stream(
        "originals/abc/source.mp4",
        _agen([b"hello", b" ", b"world"]),
        content_type="video/mp4",
    )
    assert key == "originals/abc/source.mp4"
    assert await storage.exists(key)
    path = storage.get_local_path(key)
    assert path.read_bytes() == b"hello world"
    assert storage.get_url(key) == "http://testserver/media/originals/abc/source.mp4"


@pytest.mark.asyncio
async def test_save_file_and_delete(
    settings: Settings, storage: LocalFileStorage, tmp_path: Path
) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"payload")
    key = await storage.save_file("x/y.bin", source)
    assert (settings.storage_root / "x" / "y.bin").read_bytes() == b"payload"
    await storage.delete(key)
    assert not await storage.exists(key)


@pytest.mark.asyncio
async def test_path_traversal_rejected(storage: LocalFileStorage) -> None:
    with pytest.raises(StorageError):
        storage._resolve_key("../outside.txt")
