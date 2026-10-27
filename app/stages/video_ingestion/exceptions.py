from app.core.exceptions import AppError


class VideoIngestionError(AppError):
    """Base error for the video ingestion stage."""


class UnsupportedFormatError(VideoIngestionError):
    """Raised when the input video format is not supported."""


class VideoUnavailableError(VideoIngestionError):
    """Raised when the video cannot be downloaded or opened."""


class VideoTooLargeError(VideoIngestionError):
    """Raised when the video exceeds the configured size limit."""


class MetadataExtractionError(VideoIngestionError):
    """Raised when ffprobe / metadata parsing fails."""
