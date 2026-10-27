import pytest

from app.core.config import Settings
from app.stages.video_ingestion.exceptions import UnsupportedFormatError, VideoTooLargeError
from app.stages.video_ingestion.validation import (
    extension_from_url,
    normalize_extension,
    validate_container_magic,
    validate_extension,
    validate_mime_type,
    validate_size,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(max_upload_size_mb=1)


def test_normalize_extension() -> None:
    assert normalize_extension("clip.MP4") == ".mp4"
    assert normalize_extension("a/b/c.Mov") == ".mov"
    assert normalize_extension(None) == ""


def test_extension_from_url() -> None:
    assert extension_from_url("https://cdn.example.com/v/file.mp4?token=1") == ".mp4"


def test_validate_extension_rejects_avi(settings: Settings) -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_extension(".avi", settings)


def test_validate_mime_type_rejects_unknown(settings: Settings) -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_mime_type("video/x-msvideo", settings)


def test_validate_mime_type_allows_octet_stream(settings: Settings) -> None:
    validate_mime_type("application/octet-stream", settings)


def test_validate_size(settings: Settings) -> None:
    validate_size(1024, settings)
    with pytest.raises(VideoTooLargeError):
        validate_size(settings.max_upload_size_bytes + 1, settings)


def test_validate_container_magic_mp4() -> None:
    header = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00" + b"\x00" * 32
    validate_container_magic(header, ".mp4")


def test_validate_container_magic_rejects_garbage() -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_container_magic(b"not-a-video-file" + b"\x00" * 48, ".mp4")
