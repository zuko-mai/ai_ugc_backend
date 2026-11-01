from app.core.exceptions import AppError


class HookTextEngineError(AppError):
    """Base error for the Hook Text Engine stage."""


class VideoNotFoundError(HookTextEngineError):
    """Raised when the source video cannot be resolved."""


class InvalidIntervalError(HookTextEngineError):
    """Raised when the extraction interval is invalid."""


class ExtractionError(HookTextEngineError):
    """Raised when OCR/ASR extraction fails."""


class AdaptationError(HookTextEngineError):
    """Raised when hook adaptation fails."""


class SessionNotFoundError(HookTextEngineError):
    """Raised when a stored extraction/adaptation session is missing."""


class VariantNotFoundError(HookTextEngineError):
    """Raised when a selected variant_id is unknown."""
