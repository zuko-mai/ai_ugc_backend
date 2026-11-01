from app.core.exceptions import AppError


class FaceReplacementError(AppError):
    """Base error for the face replacement stage."""


class ImageFetchError(FaceReplacementError):
    """Raised when an input image cannot be downloaded or opened."""


class ImageTooLargeError(FaceReplacementError):
    """Raised when an input image exceeds the configured size limit."""


class ImageResolutionError(FaceReplacementError):
    """Raised when an input image is below the minimum resolution."""


class ImageDecodeError(FaceReplacementError):
    """Raised when image bytes cannot be decoded."""


class ProviderNotConfiguredError(FaceReplacementError):
    """Raised when the selected model provider is missing credentials or setup."""


class ProviderTimeoutError(FaceReplacementError):
    """Raised when the model provider call times out."""


class ProviderApiError(FaceReplacementError):
    """Raised when the model provider returns an API/transport error."""


class GenerationFailedError(FaceReplacementError):
    """Raised when the provider returns an unusable or empty result."""


class FrameNotSuitableError(FaceReplacementError):
    """Raised when the generated frame fails first-frame readiness checks."""


class GenerationNotFoundError(FaceReplacementError):
    """Raised when a stored generation id cannot be resolved."""


class MissingFaceReferenceError(FaceReplacementError):
    """Raised when neither a face upload nor a reusable face URL is provided."""
