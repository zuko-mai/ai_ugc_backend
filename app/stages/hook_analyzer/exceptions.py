from app.core.exceptions import AppError


class HookAnalyzerError(AppError):
    """Base error for the hook analyzer stage."""


class VideoNotFoundError(HookAnalyzerError):
    """Raised when the source video cannot be resolved."""


class InvalidIntervalError(HookAnalyzerError):
    """Raised when the hook interval is invalid."""


class SessionNotFoundError(HookAnalyzerError):
    """Raised when cached analysis/session data is missing."""


class FrameNotFoundError(HookAnalyzerError):
    """Raised when a selected frame_id is unknown for the video."""


class FrameExtractionError(HookAnalyzerError):
    """Raised when frames cannot be extracted from the video."""


class GeminiMotionPromptError(HookAnalyzerError):
    """Raised when Gemini fails to produce a Kling motionPrompt."""
