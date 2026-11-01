"""Music Overlay — independent stage that mixes a music bed onto a finished video.

The stage does not know how the video was produced. It accepts asset URLs
(`project_id`, `asset_id`, `video_url`, `audio_url`) and returns a new asset
for the next pipeline stage.
"""

from app.stages.music_overlay.router import router

__all__ = ["router"]
