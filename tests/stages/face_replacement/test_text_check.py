import cv2
import numpy as np
import pytest

from app.stages.face_replacement.text_check import (
    check_residual_overlay_text,
    _local_text_heuristic,
)


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def test_local_heuristic_clean_image() -> None:
    img = np.full((480, 480, 3), 90, dtype=np.uint8)
    cv2.circle(img, (240, 180), 60, (160, 140, 120), -1)
    result = _local_text_heuristic(_encode(img))
    assert result.method == "local"
    # Soft gradient / face disk should not look like captions
    assert result.residual_text_likely is False or result.confidence < 0.55


def test_local_heuristic_detects_caption_bar() -> None:
    img = np.full((640, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (30, 560), (610, 620), (255, 255, 255), -1)
    cv2.putText(
        img,
        "EVERYONE IS TALKING ABOUT THIS",
        (50, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    result = _local_text_heuristic(_encode(img))
    assert result.residual_text_likely is True
    assert result.confidence >= 0.45


@pytest.mark.asyncio
async def test_check_without_api_key_uses_local() -> None:
    img = np.full((320, 320, 3), 40, dtype=np.uint8)
    result = await check_residual_overlay_text(_encode(img), api_key="")
    assert result.method == "local"
