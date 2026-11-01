from app.stages.video_ingestion.social import (
    is_instagram_reel_url,
    is_social_video_url,
    is_tiktok_url,
)


def test_detects_instagram_reel_urls() -> None:
    assert is_instagram_reel_url("https://www.instagram.com/reel/ABC123/")
    assert is_instagram_reel_url("https://instagram.com/reels/ABC123/?igsh=x")
    assert is_instagram_reel_url("https://www.instagram.com/p/ABC123/")
    assert not is_instagram_reel_url("https://example.com/video.mp4")
    assert not is_instagram_reel_url("https://www.instagram.com/user/")


def test_detects_tiktok_urls() -> None:
    assert is_tiktok_url("https://www.tiktok.com/@user/video/7123456789012345678")
    assert is_tiktok_url("https://vm.tiktok.com/ZMabcdef/")
    assert is_tiktok_url("https://vt.tiktok.com/ZSxyz/")
    assert not is_tiktok_url("https://www.tiktok.com/@user")
    assert not is_tiktok_url("https://example.com/video.mp4")


def test_social_video_url_covers_ig_and_tt() -> None:
    assert is_social_video_url("https://www.instagram.com/reel/ABC123/")
    assert is_social_video_url("https://www.tiktok.com/@u/video/1")
    assert not is_social_video_url("https://cdn.example.com/file.mp4")
