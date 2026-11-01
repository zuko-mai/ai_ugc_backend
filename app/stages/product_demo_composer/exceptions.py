from app.core.exceptions import AppError


class ProductDemoComposerError(AppError):
    """Base error for the product demo composer stage."""


class VideoFetchError(ProductDemoComposerError):
    """Raised when a source video cannot be downloaded or opened."""


class VideoTooLargeError(ProductDemoComposerError):
    """Raised when a source video exceeds the configured size limit."""


class ProbeError(ProductDemoComposerError):
    """Raised when ffprobe cannot read a media file."""


class ComposeError(ProductDemoComposerError):
    """Raised when ffmpeg cannot compose the output video."""


class InvalidSettingsError(ProductDemoComposerError):
    """Raised when compose settings are inconsistent with the inputs."""


class UnsupportedFormatError(ProductDemoComposerError):
    """Raised when an uploaded demo video has an unsupported container/MIME."""


class JobNotFoundError(ProductDemoComposerError):
    """Raised when a stored job id is unknown."""
