"""Video generation provider adapters."""

from app.stages.hook_video_generator.providers.base import (
    ProviderImage,
    VideoGenerationProvider,
    VideoProviderPoll,
    VideoProviderRequest,
    VideoProviderSubmission,
    normalize_remote_status,
)
from app.stages.hook_video_generator.providers.kling import KlingVideoProvider
from app.stages.hook_video_generator.providers.registry import (
    get_provider,
    list_providers,
    register_provider,
)

__all__ = [
    "KlingVideoProvider",
    "ProviderImage",
    "VideoGenerationProvider",
    "VideoProviderPoll",
    "VideoProviderRequest",
    "VideoProviderSubmission",
    "get_provider",
    "list_providers",
    "normalize_remote_status",
    "register_provider",
]
