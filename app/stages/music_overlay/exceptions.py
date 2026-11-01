from app.core.exceptions import AppError


class MusicOverlayError(AppError):
    """Base error for the music overlay stage."""


class MediaFetchError(MusicOverlayError):
    """Raised when a source video or music file cannot be opened."""


class MediaTooLargeError(MusicOverlayError):
    """Raised when a source file exceeds the configured size limit."""


class ProbeError(MusicOverlayError):
    """Raised when ffprobe cannot read a media file."""


class OverlayError(MusicOverlayError):
    """Raised when ffmpeg cannot mix the music bed onto the video."""


class InvalidSettingsError(MusicOverlayError):
    """Raised when overlay settings are inconsistent with the media."""


class UnsupportedFormatError(MusicOverlayError):
    """Raised when an uploaded music file has an unsupported type."""


class JobNotFoundError(MusicOverlayError):
    """Raised when a stored job id is unknown."""
