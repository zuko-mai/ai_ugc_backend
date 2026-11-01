from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings


def _write_jpeg(path: Path, *, width: int, height: int, with_caption: bool = False) -> None:
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 3), min(width, height) // 6, (180, 160, 140), -1)
    if with_caption:
        cv2.rectangle(
            img,
            (20, height - 70),
            (width - 20, height - 20),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            img,
            "WAIT FOR IT",
            (40, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    assert cv2.imwrite(str(path), img)


def _store_image(settings: Settings, relative_key: str, path: Path) -> str:
    dest = settings.storage_root / relative_key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
    return f"{settings.public_base_url}/media/{relative_key}"


@pytest.fixture
def frame_and_face(settings: Settings, tmp_path: Path) -> tuple[str, Path]:
    frame_path = tmp_path / "frame.jpg"
    face_path = tmp_path / "face.jpg"
    _write_jpeg(frame_path, width=640, height=640, with_caption=True)
    _write_jpeg(face_path, width=320, height=320, with_caption=False)
    frame_url = _store_image(settings, "test_inputs/frame.jpg", frame_path)
    return frame_url, face_path


def _generate(
    client: TestClient,
    *,
    video_id: str,
    frame_url: str,
    face_path: Path,
    **extra: object,
):
    data = {
        "video_id": video_id,
        "selected_frame_url": frame_url,
        **{k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in extra.items()},
    }
    with face_path.open("rb") as fh:
        return client.post(
            "/v1/face-replacement/generate",
            data=data,
            files={"face_reference": ("face.jpg", fh, "image/jpeg")},
        )


def test_generate_face_replacement_stub(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    video_id = str(uuid4())

    response = _generate(
        client,
        video_id=video_id,
        frame_url=frame_url,
        face_path=face_path,
        remove_overlay_text=True,
        preserve_original_composition=True,
        preserve_pose=True,
        preserve_lighting=True,
        model_provider="stub",
        model_name="stub-passthrough",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == video_id
    assert body["status"] == "success"
    assert body["width"] == 640
    assert body["height"] == 640
    assert body["model_provider"] == "stub"
    assert body["model_name"] == "stub-passthrough"
    assert body["text_removal_requested"] is True
    assert body["generated_frame_url"].startswith("http://testserver/media/face_replacement/")
    assert body["generation_id"]
    assert isinstance(body["warnings"], list)
    assert any("residual_overlay_text" in w or "stub_provider" in w for w in body["warnings"])

    usage = body["usage"]
    assert usage["total_tokens"] == 0
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["currency"] == "USD"
    assert usage["processing_time_seconds"] >= 0
    assert usage["provider_latency_seconds"] is not None
    assert usage["provider_latency_seconds"] >= 0

    media = client.get(body["generated_frame_url"].removeprefix("http://testserver"))
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")


def test_regeneration_creates_new_generation(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    video_id = str(uuid4())

    first = _generate(
        client,
        video_id=video_id,
        frame_url=frame_url,
        face_path=face_path,
        remove_overlay_text=False,
        model_provider="stub",
    )
    second = _generate(
        client,
        video_id=video_id,
        frame_url=frame_url,
        face_path=face_path,
        remove_overlay_text=False,
        model_provider="stub",
        preserve_lighting=False,
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["generation_id"] != second.json()["generation_id"]
    assert first.json()["generated_frame_url"] != second.json()["generated_frame_url"]


def test_low_resolution_frame_rejected(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    tiny = tmp_path / "tiny.jpg"
    face = tmp_path / "face_ok.jpg"
    _write_jpeg(tiny, width=32, height=32)
    _write_jpeg(face, width=128, height=128)
    tiny_url = _store_image(settings, "test_inputs/tiny.jpg", tiny)

    response = _generate(
        client,
        video_id=str(uuid4()),
        frame_url=tiny_url,
        face_path=face,
        model_provider="stub",
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "image_resolution_too_low"


def test_low_resolution_face_upload_rejected(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    frame = tmp_path / "frame_ok.jpg"
    tiny_face = tmp_path / "tiny_face.jpg"
    _write_jpeg(frame, width=128, height=128)
    _write_jpeg(tiny_face, width=32, height=32)
    frame_url = _store_image(settings, "test_inputs/frame_ok.jpg", frame)

    response = _generate(
        client,
        video_id=str(uuid4()),
        frame_url=frame_url,
        face_path=tiny_face,
        model_provider="stub",
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "image_resolution_too_low"


def test_unknown_provider_rejected(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    response = _generate(
        client,
        video_id=str(uuid4()),
        frame_url=frame_url,
        face_path=face_path,
        model_provider="does-not-exist",
    )
    assert response.status_code == 503
    assert response.json()["error_code"] == "provider_not_configured"


def test_missing_frame_returns_404(
    client: TestClient, frame_and_face: tuple[str, Path]
) -> None:
    _, face_path = frame_and_face
    response = _generate(
        client,
        video_id=str(uuid4()),
        frame_url="http://testserver/media/missing/frame.jpg",
        face_path=face_path,
        model_provider="stub",
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "image_fetch_failed"


def test_missing_face_file_rejected(client: TestClient, frame_and_face: tuple[str, Path]) -> None:
    frame_url, _ = frame_and_face
    response = client.post(
        "/v1/face-replacement/generate",
        data={
            "video_id": str(uuid4()),
            "selected_frame_url": frame_url,
            "model_provider": "stub",
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "missing_face_reference"


def test_originals_archived_unchanged(
    client: TestClient, settings: Settings, frame_and_face: tuple[str, Path]
) -> None:
    frame_url, face_path = frame_and_face
    video_id = uuid4()
    original_bytes = (settings.storage_root / "test_inputs/frame.jpg").read_bytes()
    face_bytes = face_path.read_bytes()

    response = _generate(
        client,
        video_id=str(video_id),
        frame_url=frame_url,
        face_path=face_path,
        model_provider="stub",
    )
    assert response.status_code == 200, response.text

    assert (settings.storage_root / "test_inputs/frame.jpg").read_bytes() == original_bytes

    inputs_dir = settings.storage_root / "face_replacement" / str(video_id) / "inputs"
    frame_files = [
        p
        for p in inputs_dir.glob("selected_frame_*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    face_files = [
        p
        for p in inputs_dir.glob("face_reference_*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert frame_files
    assert face_files
    assert frame_files[0].read_bytes() == original_bytes
    assert face_files[0].read_bytes() == face_bytes
