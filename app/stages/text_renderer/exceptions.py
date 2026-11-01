from app.core.exceptions import AppError


class TextRendererError(AppError):
    """Base error for the text renderer stage."""


class VideoFetchError(TextRendererError):
    """Raised when the source video cannot be downloaded or opened."""


class VideoTooLargeError(TextRendererError):
    """Raised when the source video exceeds the configured size limit."""


class ProbeError(TextRendererError):
    """Raised when ffprobe cannot read a media file."""


class RenderError(TextRendererError):
    """Raised when ffmpeg cannot burn text into the video."""


class FontNotFoundError(TextRendererError):
    """Raised when no usable font file is available for the requested family."""


class InvalidTextError(TextRendererError):
    """Raised when text content is unsafe or malformed for deterministic burn-in."""


class RenderNotFoundError(TextRendererError):
    """Raised when a stored render id is unknown."""
