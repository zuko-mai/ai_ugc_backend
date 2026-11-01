from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ai-ugc-creator"
    app_env: str = "development"
    log_level: str = "INFO"

    storage_root: Path = Field(default=Path("./data/storage"))
    public_base_url: str = "http://localhost:8000"

    max_upload_size_mb: int = 500
    download_timeout_seconds: int = 120

    allowed_extensions: str = ".mp4,.mov"
    allowed_mime_types: str = "video/mp4,video/quicktime"

    google_ai_api_key: str = ""
    # Legacy shared default (still used where a stage has no dedicated model).
    gemini_model: str = "gemini-3.8-flash"
    # Cheap video-capable model for hook motionPrompt (select frame + motion).
    hook_analyzer_motion_model: str = "gemini-3.1-flash-lite"
    # Single-call Hook Text (video → OCR/ASR + variants).
    hook_text_model: str = "gemini-3.1-flash-lite"
    hook_text_fallback_model: str = "gemini-2.5-flash"
    # Text-only adaptation (no video) — cheapest working flash-lite.
    gemini_text_model: str = "gemini-2.5-flash-lite"

    # Face Replacement stage
    face_replacement_provider: str = "gemini"
    # Nano Banana Pro (default); secondary option: gemini-3.1-flash-image (Nano Banana 2)
    face_replacement_model: str = "gemini-3-pro-image"
    face_replacement_timeout_seconds: int = 300
    max_face_image_size_mb: int = 20
    min_selected_frame_width: int = 512
    min_selected_frame_height: int = 512
    min_face_reference_width: int = 256
    min_face_reference_height: int = 256

    # Hook Video Generator stage (Kling 3.0 image-to-video)
    hook_video_provider: str = "kling"
    hook_video_model: str = "kling-3.0"
    hook_video_timeout_seconds: int = 300
    hook_video_min_duration_seconds: float = 2.0
    hook_video_max_duration_seconds: float = 4.0
    hook_video_duration_tolerance_seconds: float = 0.25
    hook_video_http_submit_url: str = ""
    hook_video_http_status_url: str = ""
    hook_video_http_api_key: str = ""
    hook_video_http_auth_header: str = "Authorization"
    hook_video_http_auth_scheme: str = "Bearer"
    max_hook_frame_size_mb: int = 20

    # Text Renderer stage
    max_text_render_video_size_mb: int = 200

    # Product Demo Composer stage
    max_product_demo_video_size_mb: int = 500

    # Music Overlay stage
    max_music_overlay_video_size_mb: int = 500
    max_music_overlay_audio_size_mb: int = 100

    # Kling 3.0 image-to-video (General pack; not Kling 4.0)
    kling_api_key: str = ""
    kling_api_base_url: str = "https://api-singapore.klingai.com"
    kling_audio: str = "off"
    kling_multi_shot: bool = False
    kling_prefer_base64: bool = False

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def extension_set(self) -> set[str]:
        return {
            ext.strip().lower()
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def mime_type_set(self) -> set[str]:
        return {
            mime.strip().lower()
            for mime in self.allowed_mime_types.split(",")
            if mime.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
