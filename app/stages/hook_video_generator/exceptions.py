from app.core.exceptions import AppError


class HookVideoError(AppError):
    """Base error for the hook video generator stage."""


class PayloadFetchError(HookVideoError):
    """Raised when bytes cannot be read from storage or a remote URL."""


class FrameFetchError(PayloadFetchError):
    """Raised when the first frame cannot be downloaded or opened."""


class FrameTooLargeError(HookVideoError):
    """Raised when the first frame exceeds the configured size limit."""


class FrameDecodeError(HookVideoError):
    """Raised when first-frame bytes cannot be decoded as an image."""


class ProviderNotConfiguredError(HookVideoError):
    """Raised when the selected model provider is missing credentials or setup."""


class ProviderTimeoutError(HookVideoError):
    """Raised when a provider call exceeds its timeout."""


class ProviderApiError(HookVideoError):
    """Raised when a provider returns an API or transport error."""


class GenerationFailedError(HookVideoError):
    """Raised when a task cannot be queued."""


class JobNotFoundError(HookVideoError):
    """Raised when a generation job id is unknown."""


class ComparisonNotFoundError(HookVideoError):
    """Raised when a model-comparison id is unknown."""


class ProbeError(HookVideoError):
    """Raised when ffprobe cannot read a media file."""
