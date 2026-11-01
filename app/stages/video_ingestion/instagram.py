"""Backward-compatible re-exports for Instagram helpers. """

from app.stages.video_ingestion.social import (
    download_instagram_reel,
    download_social_video,
    is_instagram_reel_url,
    is_social_video_url,
    is_tiktok_url,
    social_platform,
)

__all__ = [
    "download_instagram_reel",
    "download_social_video",
    "is_instagram_reel_url",
    "is_social_video_url",
    "is_tiktok_url",
    "social_platform",
]
